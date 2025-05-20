import argparse
import os
import tarfile
import zipfile
from pathlib import Path
from typing import List

from tqdm import tqdm

SUPPORTED_EXTS = [
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
]


def is_archive(path: Path) -> bool:
    return any(str(path).lower().endswith(ext) for ext in SUPPORTED_EXTS)


def extract_zip(fp: Path, dest: Path):
    with zipfile.ZipFile(fp) as zf:
        members = zf.infolist()
        for m in tqdm(members, desc=f"unzipping {fp.name}", unit="file"):
            zf.extract(m, path=dest)


def extract_tar(fp: Path, dest: Path):
    mode = "r"
    if fp.suffixes[-2:] == [".tar", ".gz"] or fp.suffix == ".tgz":
        mode = "r:gz"
    elif fp.suffixes[-2:] == [".tar", ".bz2"]:
        mode = "r:bz2"
    with tarfile.open(fp, mode) as tf:
        members = tf.getmembers()
        for m in tqdm(members, desc=f"untarring {fp.name}", unit="file"):
            tf.extract(m, path=dest)


def extract_archive(fp: Path, dest_root: Path):
    dest_dir = dest_root / fp.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {fp} → {dest_dir}")
    if fp.suffix == ".zip":
        extract_zip(fp, dest_dir)
    else:
        extract_tar(fp, dest_dir)
    print(f"✓ Done {fp}")


def gather_archives(target: Path, recursive: bool) -> List[Path]:
    if target.is_file() and is_archive(target):
        return [target]
    archives = []
    pattern = "**/*" if recursive else "*"
    for p in target.glob(pattern):
        if p.is_file() and is_archive(p):
            archives.append(p)
    return archives


def main():
    parser = argparse.ArgumentParser(description="Extract supported archives (zip/tar.gz/tgz/tar.bz2) to destination.")
    parser.add_argument("--path", type=str, help="Archive file or directory containing archives")
    parser.add_argument("--dest", type=str, default="extracted", help="Destination root for extracted folders")
    parser.add_argument("--recursive", action="store_true", help="Recursively search directories for archives")
    args = parser.parse_args()

    target = Path(args.path).resolve()
    dest_root = Path(args.dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    archives = gather_archives(target, args.recursive)
    if not archives:
        print("No supported archives found.")
        return

    for fp in archives:
        extract_archive(fp, dest_root)

    print("All archives extracted.")


if __name__ == "__main__":
    main() 