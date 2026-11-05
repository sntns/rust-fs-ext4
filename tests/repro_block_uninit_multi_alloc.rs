//! Block allocations into a `BLOCK_UNINIT` group, more than once per mount.
//!
//! The allocator decides "this group is uninit, treat its bitmap as all
//! free" from `Filesystem::groups`, the descriptors read at mount time.
//! The first allocation into such a group clears the flag on disk (and in
//! the transaction buffer), but the mount-time copy still says uninit: the
//! next allocation in the same mount — the next sub-run of the same
//! `apply_pwrite`, the extent-tree leaf block of that same write, or the
//! next file — gets the same "first free" blocks again. Two inodes (or two
//! extents of one inode) then share physical blocks, and the later write
//! silently overwrites the earlier one; when the overwritten block is an
//! extent-tree node, the next lookup fails with
//! `CorruptExtentTree("bad extent header magic")`.
//!
//! The images come from the host's real `mke2fs` (flex_bg: every group but
//! the flex-group head is left `BLOCK_UNINIT`), which this crate's own mkfs
//! doesn't produce. Skipped when `mke2fs` isn't installed; when `e2fsck`
//! is, it's used as the oracle as well.

use fs_ext4::block_io::FileDevice;
use fs_ext4::fs::Filesystem;
use std::collections::BTreeMap;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const MIB: u64 = 1024 * 1024;

fn mke2fs_to_tmp(tag: &str, size: u64) -> Option<String> {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let path = format!(
        "{}/fs_ext4_uninit_{tag}_{}_{n}.img",
        std::env::temp_dir().display(),
        std::process::id()
    );
    std::fs::File::create(&path).ok()?.set_len(size).ok()?;
    let out = Command::new("mke2fs")
        .args([
            "-q",
            "-F",
            "-t",
            "ext4",
            "-b",
            "4096",
            "-E",
            "lazy_itable_init=1",
        ])
        .arg(&path)
        .output();
    match out {
        Ok(o) if o.status.success() => Some(path),
        Ok(o) => panic!("mke2fs failed: {}", String::from_utf8_lossy(&o.stderr)),
        Err(_) => {
            eprintln!("skip: mke2fs not installed");
            let _ = std::fs::remove_file(&path);
            None
        }
    }
}

/// A file's content: distinct per file and per 4 KiB block, so that any
/// block landing on another one's physical block reads back wrong.
fn pattern(seed: u32, len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    while v.len() < len {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v.extend_from_slice(&x.to_le_bytes());
    }
    v.truncate(len);
    v
}

fn read_back(fs: &Filesystem, path: &str) -> Vec<u8> {
    let mut reader = |ino: u32| fs.read_inode_verified(ino).map(|(i, _)| i);
    let ino = fs_ext4::path::lookup(fs.dev.as_ref(), &fs.sb, &mut reader, path)
        .unwrap_or_else(|e| panic!("lookup {path}: {e:?}"));
    let (inode, _) = fs.read_inode_verified(ino).expect("read inode");
    fs_ext4::file_io::read_all(fs, &inode).unwrap_or_else(|e| panic!("read {path}: {e:?}"))
}

/// Every file reads back as written, the driver's verifier finds no block
/// claimed twice, and (when installed) the real e2fsck agrees.
fn assert_intact(path: &str, files: &BTreeMap<String, Vec<u8>>) {
    {
        let dev = FileDevice::open(path).expect("ro");
        let fs = Filesystem::mount(Arc::new(dev)).expect("remount");
        for (name, want) in files {
            let got = read_back(&fs, name);
            assert_eq!(got.len(), want.len(), "{name}: size");
            if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
                panic!("{name}: content differs from byte {i} (block {})", i / 4096);
            }
        }
        let report = fs_ext4::verify::verify(&fs).expect("verify");
        assert!(report.is_clean(), "verify: {:?}", report.errors);
    }
    if let Ok(o) = Command::new("e2fsck").args(["-fn", path]).output() {
        assert!(
            o.status.success(),
            "e2fsck -fn: {}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
    }
    let _ = std::fs::remove_file(path);
}

/// One write larger than any group's free run: split into sub-runs, which
/// all used to land on the same blocks of the first uninit group.
#[test]
fn one_write_spanning_several_uninit_groups() {
    let Some(img) = mke2fs_to_tmp("span", 1024 * MIB) else {
        return;
    };
    let mut files = BTreeMap::new();
    files.insert("/big".to_string(), pattern(1, (300 * MIB) as usize));
    {
        let dev = FileDevice::open_rw(&img).expect("rw");
        let fs = Filesystem::mount(Arc::new(dev)).expect("mount");
        for (name, data) in &files {
            fs.apply_create(name, 0o644).expect("create");
            fs.apply_pwrite(name, 0, data).expect("pwrite");
        }
    }
    assert_intact(&img, &files);
}

/// Many files in one mount, enough to fill one group and cross into the
/// next uninit ones: each crossing used to hand out blocks already given
/// to an earlier file (or to an extent-tree leaf).
#[test]
fn many_files_in_one_mount_crossing_uninit_groups() {
    let Some(img) = mke2fs_to_tmp("many", 1024 * MIB) else {
        return;
    };
    let mut files = BTreeMap::new();
    for i in 0..60u32 {
        let len = (5 * MIB + (i as u64 * 37 * 4096) % MIB) as usize;
        files.insert(format!("/d/f{i:03}"), pattern(100 + i, len));
    }
    {
        let dev = FileDevice::open_rw(&img).expect("rw");
        let fs = Filesystem::mount(Arc::new(dev)).expect("mount");
        fs.apply_mkdir("/d", 0o755).expect("mkdir");
        for (name, data) in &files {
            fs.apply_create(name, 0o644).expect("create");
            fs.apply_pwrite(name, 0, data).expect("pwrite");
        }
    }
    assert_intact(&img, &files);
}

/// The same, but remounting before every file, as remora-etcher does (one
/// mount per `image-cp-dir` entry): only the allocations within a single
/// write share a mount.
#[test]
fn many_files_one_mount_each_crossing_uninit_groups() {
    let Some(img) = mke2fs_to_tmp("remount", 1024 * MIB) else {
        return;
    };
    let mut files = BTreeMap::new();
    for i in 0..60u32 {
        let len = (5 * MIB + (i as u64 * 37 * 4096) % MIB) as usize;
        files.insert(format!("/d/f{i:03}"), pattern(200 + i, len));
    }
    {
        let dev = FileDevice::open_rw(&img).expect("rw");
        let fs = Filesystem::mount(Arc::new(dev)).expect("mount");
        fs.apply_mkdir("/d", 0o755).expect("mkdir");
    }
    for (name, data) in &files {
        let dev = FileDevice::open_rw(&img).expect("rw");
        let fs = Filesystem::mount(Arc::new(dev)).expect("mount");
        fs.apply_create(name, 0o644).expect("create");
        fs.apply_pwrite(name, 0, data).expect("pwrite");
    }
    assert_intact(&img, &files);
}
