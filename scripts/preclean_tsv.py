# scripts/preclean_tsv.py
import argparse, multiprocessing as mp, os, json
from pathlib import Path
from tqdm import tqdm
import torch, torchaudio
import soundfile as sf

def _test_file(args):
    root, rel_path = args
    full = root / rel_path
    if not full.is_file():
        return None             # missing
    try:
        torchaudio.load(full)
        return rel_path         # good
    except Exception:
        try:
            sf.read(full)       # 2nd try
            return rel_path     # good
        except Exception:
            return None         # corrupt

def clean_split(tsv_path: Path, root: Path):
    lines = [l.rstrip().split("\t") for l in tsv_path.read_text().splitlines() if l.strip()]
    pool = mp.Pool(mp.cpu_count())
    good = []
    bad  = []

    for rel, ok in zip(lines, tqdm(pool.imap(_test_file, [(root, l[0]) for l in lines]),
                                   total=len(lines), desc=f"Checking {tsv_path.name}")):
        if ok:
            good.append(rel)
        else:
            bad.append(rel[0])
    pool.close(); pool.join()

    clean_path = tsv_path.with_name(tsv_path.stem + "_clean.tsv")
    with clean_path.open("w") as f:
        for p, dur in good:
            f.write(f"{p}\t{dur}\n")

    log_path = tsv_path.with_name("removed.txt")
    log_path.write_text("\n".join(bad))

    print(f"{tsv_path.name}: kept {len(good)}, removed {len(bad)} ➜ {clean_path}")
    return clean_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", help="data/processed/<dataset_name>")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    root = Path(args.dataset_dir)
    for split in args.splits:
        tsv = root / f"{split}.tsv"
        if tsv.exists():
            clean_split(tsv, root)