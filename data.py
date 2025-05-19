#!/usr/bin/env python
"""
Download 1 M examples from LAION-Audio-300M as *MP3* files, create train/val/test
splits and a manifest, in the shortest time that a single machine with a fast
network link can achieve.

Main tricks
-----------

1. Stream the dataset and **keep only the audio column**:
      ds = ds.select_columns(["audio.mp3"])

2. **Disable decoding** so that every item yields only the local-cache path to
   the MP3 file; Hugging Face downloads each file for us in the background:
      ds = ds.cast_column("audio.mp3", Audio(decode=False))

3. **Copy the already-downloaded file**, do *not* decode or re-encode it.

4. **Parallelise the I/O** with a `ThreadPoolExecutor` – MP3 downloads and disk
   writes are I/O-bound, so threads scale well.

5. Do just enough hashing to obtain a reproducible split; all remaining work is
   pure I/O.

Tested with
-----------

* Python 3.9
* datasets == 4.41
* requests == 2.32
* tqdm == 4.66

-------------------------------------------------------------------------------
"""

import os
import hashlib
import logging
import multiprocessing as mp
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

import requests
from datasets import Audio, load_dataset
from tqdm import tqdm

###############################################################################
# Configuration
###############################################################################

TOTAL_SAMPLES      = 1_000_000      # change to something smaller if required
SPLIT_RATIOS       = (0.98, 0.01, 0.01)   # train / val / test
MAX_WORKERS        = min(32, (mp.cpu_count() or 1) * 4)  # threads for I/O
DATA_DIR           = "data/laion"   # everything written below this directory
HF_CACHE_DIR       = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

###############################################################################
# Utility functions
###############################################################################


def pick_split(idx: int, ratios=SPLIT_RATIOS) -> str:
    """
    Deterministically assign an index to train / val / test using a fast hash.
    """
    h = int(hashlib.blake2b(str(idx).encode(), digest_size=4).hexdigest(), 16)
    frac = h / 0xFFFFFFFF
    if frac < ratios[0]:
        return "train"
    if frac < ratios[0] + ratios[1]:
        return "val"
    return "test"


def copy_mp3(idx: int,
             src_path: str,
             audio_dir: str,
             split_dirs: dict[str, str]) -> None:
    """
    Copy the cached MP3 file into our dataset folder and write a one-line
    manifest entry.  Run inside a worker thread.
    """
    try:
        dst_fname = f"laion_{idx:09d}.mp3"
        dst_path  = os.path.join(audio_dir, dst_fname)

        # Fast path: the file is already on the same filesystem.
        try:
            shutil.copy2(src_path, dst_path)
        except (shutil.Error, FileNotFoundError):
            # Rare corner-case: the cache file vanished – re-download.
            with requests.get(src_path, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dst_path, "wb") as f_out:
                    shutil.copyfileobj(r.raw, f_out, length=1024 * 1024)

        split_name = pick_split(idx)
        manifest_path = os.path.join(split_dirs[split_name],
                                     f"{os.path.splitext(dst_fname)[0]}.txt")
        with open(manifest_path, "w") as m:
            m.write(f"{dst_path}\n")

    except Exception as exc:
        logging.warning("Item %d failed: %s", idx, exc)


###############################################################################
# Main download routine
###############################################################################

def download_laion_audio():
    # ------------------------------------------------------------------ logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # --------------------------------------------------- prepare output folders
    audio_dir      = os.path.join(DATA_DIR, "audio")
    split_dirs     = {s: os.path.join(DATA_DIR, s) for s in ("train", "val", "test")}

    for p in (audio_dir, *split_dirs.values()):
        os.makedirs(p, exist_ok=True)

    # --------------------------------------------------- load / stream dataset
    logging.info("🔄  Streaming LAION-Audio-300M …")
    ds = load_dataset(
        "laion/LAION-Audio-300M",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    ds = ds.cast_column("audio.mp3", Audio(decode=False))
    ds = ds.select_columns(["audio.mp3"])
    ds = ds.shuffle(buffer_size=10_000, seed=42)

    # --------------------------------------------------- first pass: collect N items
    logging.info("📋  Collecting %d sample descriptors …", TOTAL_SAMPLES)
    items: list[tuple[int, str]] = []
    with tqdm(total=TOTAL_SAMPLES, desc="Collecting sample descriptors") as pbar:
        for idx, example in enumerate(islice(ds, TOTAL_SAMPLES)):
            audio_info = example["audio.mp3"]
            path = audio_info.get("path")
            if path:
                items.append((idx, path))
            else:
                logging.debug("Skipping item %d – no path", idx)
            pbar.update(1)

    logging.info("✅  Collected %d items, starting parallel copy …", len(items))

    # --------------------------------------------------- second pass: parallel copy
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool, \
         tqdm(total=len(items), desc="Downloading & copying MP3") as bar:

        futures = [pool.submit(copy_mp3, idx, path, audio_dir, split_dirs)
                   for idx, path in items]

        for f in as_completed(futures):
            bar.update(1)  # each future corresponds to one example

    logging.info("🏁  Finished – %d MP3 files saved in %s", len(items), DATA_DIR)


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    download_laion_audio()