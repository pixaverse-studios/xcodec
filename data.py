import os, tarfile, shutil, requests, concurrent.futures as cf
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# --------------------------------------------------------------------------
DATA_DIR       = Path("data")
TOTAL_SAMPLES  = 2_000_000        # 2 million samples
MAX_WORKERS    = 64
CHUNK          = 1 << 20          # 1 MiB
SPLIT_RATIOS   = (0.98, 0.01, 0.01)
# --------------------------------------------------------------------------

# one global HTTP session
SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS*2,
                                        pool_maxsize=MAX_WORKERS*2,
                                        max_retries=3)
SESSION.mount("http://", ADAPTER)
SESSION.mount("https://", ADAPTER)


def idx_to_split(i: int) -> str:
    r = hash(str(i)) / 2**32
    return "train" if r < SPLIT_RATIOS[0] else \
           "val"   if r < sum(SPLIT_RATIOS[:2]) else "test"


def download_one(args):
    """Download __url__ to local disk –   returns (ok, idx, path)."""
    idx, item, audio_dir = args
    try:
        url = item["__url__"]                # <- fast path
        ext = os.path.splitext(url)[1] or ".mp3"
        tgt = audio_dir / f"laion_{idx:09d}{ext}"

        if tgt.exists():                     # already there
            return True, idx, tgt

        r = SESSION.get(url, stream=True, timeout=30)
        r.raise_for_status()
        tmp = tgt.with_suffix(tgt.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
        tmp.rename(tgt)
        return True, idx, tgt
    except Exception:
        return False, idx, None


def download_laion():
    print("⇣  streaming metadata …")
    ds = load_dataset("laion/LAION-Audio-300M",
                      split="train", streaming=True)

    laion_root = DATA_DIR / "laion"
    audio_dir  = laion_root / "audio"
    split_dirs = {s: laion_root / s for s in ("train", "val", "test")}
    audio_dir.mkdir(parents=True, exist_ok=True)
    for p in split_dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    done = 0
    with cf.ThreadPoolExecutor(MAX_WORKERS) as pool, \
         tqdm(total=TOTAL_SAMPLES, unit="file") as bar:

        futures = set()
        for idx, item in enumerate(ds):
            if done >= TOTAL_SAMPLES:
                break

            futures.add(pool.submit(download_one, (idx, item, audio_dir)))

            while len(futures) >= MAX_WORKERS * 4:
                finished, futures = cf.wait(futures,
                                            return_when=cf.FIRST_COMPLETED)
                for fut in finished:
                    ok, i, path = fut.result()
                    if ok:
                        split = idx_to_split(i)
                        (split_dirs[split] / f"{path.stem}.txt").write_text(
                            str(path) + "\n"
                        )
                        done += 1
                        bar.update(1)
                    if done >= TOTAL_SAMPLES:
                        break

        # drain remaining
        for fut in cf.as_completed(futures):
            ok, i, path = fut.result()
            if ok:
                split = idx_to_split(i)
                (split_dirs[split] / f"{path.stem}.txt").write_text(
                    str(path) + "\n"
                )
                done += 1
                bar.update(1)
            if done >= TOTAL_SAMPLES:
                break

    print(f"\n✅  downloaded {done:,} MP3 files to {audio_dir}")


# --------------------------------------------------------------------------
# LibriSpeech (unchanged)
# --------------------------------------------------------------------------
def download_librispeech():
    url      = "https://www.openslr.org/resources/12/train-other-500.tar.gz"
    tar_path = DATA_DIR / "train-other-500.tar.gz"

    if not tar_path.exists():
        with SESSION.get(url, stream=True) as r, tar_path.open("wb") as f, \
             tqdm(total=int(r.headers.get("content-length", 0)),
                  unit="iB", unit_scale=True, desc="train-other-500.tar.gz") as bar:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print("⇡  extracting LibriSpeech …")
    with tarfile.open(tar_path, bufsize=CHUNK) as tar:
        for member in tqdm(tar.getmembers(), unit="file"):
            tar.extract(member, DATA_DIR)

    tar_path.unlink(missing_ok=True)
    print("🎉  LibriSpeech ready")


if __name__ == "__main__":
    download_laion()          # fast LAION downloader
    download_librispeech()    # comment this line if you don't need it