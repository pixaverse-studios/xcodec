import os
import json
import shutil
import argparse
from pathlib import Path
import concurrent.futures
import random

from tqdm import tqdm

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"[WARN] Could not copy {src} -> {dst}: {e}")


def resolve_audio_path(entry_path: str, base_dir: Path) -> Path:
    """Return absolute Path to audio.
    If entry_path is already absolute, leave as is; otherwise join to *base_dir*.
    """
    p = Path(entry_path)
    if p.is_absolute():
        return p
    # Remove potential leading './'
    return (base_dir / entry_path.lstrip("./")).resolve()


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reorganise an already extracted processed dataset into Data/train/audios etc."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Directory that contains train.json, val.json and all segment folders.",
    )
    parser.add_argument(
        "--dest_root",
        type=str,
        default="Data",
        help="Destination base directory (will create train/audios and val/audios).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Dataset tag to prefix output filenames (e.g., 'rhymes' or 'gopipe').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Thread workers for copying",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.is_dir():
        print(f"[ERROR] root_dir {root_dir} is not a directory")
        return

    train_json = root_dir / "train.json"
    val_json = root_dir / "val.json"

    if train_json.exists() and val_json.exists():
        # Normal path – read lists of dicts
        with open(train_json) as f:
            train_data = json.load(f)
        with open(val_json) as f:
            val_data = json.load(f)
        json_mode = True
    else:
        print("[WARN] train.json / val.json not found – scanning directory tree for audio.wav files …")
        exts = ("*.wav", "*.flac", "*.mp3")
        audio_paths = []
        for pattern in exts:
            audio_paths.extend(root_dir.rglob(pattern))
        if not audio_paths:
            print("[ERROR] No audio.wav files discovered under", root_dir)
            return

        random.shuffle(audio_paths)
        split_idx = int(len(audio_paths) * 0.9)
        train_data = audio_paths[:split_idx]
        val_data = audio_paths[split_idx:]
        json_mode = False
        print(f"Total audio files: {len(audio_paths)} → train {len(train_data)}, val {len(val_data)}")

    dest_root = Path(args.dest_root).resolve()
    train_audio_dir = dest_root / "train" / "audios"
    val_audio_dir = dest_root / "val" / "audios"

    def _copy(entry, split_dir):
        if json_mode:
            abs_src = resolve_audio_path(entry["audio_segment_file_path"], root_dir)
        else:
            abs_src = entry  # already Path
        try:
            # Try to derive identifiers – works for both id/segment/audio.wav and id/audio_x.wav
            video_id = abs_src.parent.name  # id folder
            maybe_segment = abs_src.stem  # filename without ext
            if maybe_segment.startswith("segment_"):
                # original pattern id/segment_xxxx/audio.wav
                segment_name = maybe_segment  # segment_xxxx
            else:
                segment_name = maybe_segment  # audio_1 etc.
            dst_name = f"{args.tag}_{video_id}_{segment_name}{abs_src.suffix}"
        except Exception:
            dst_name = f"{args.tag}_{abs_src.name}"
        dst = split_dir / dst_name
        safe_copy(abs_src, dst)

    print(f"Copying {len(train_data)} train + {len(val_data)} val audio files → {dest_root} …")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(_copy, e, train_audio_dir) for e in train_data
        ] + [
            ex.submit(_copy, e, val_audio_dir) for e in val_data
        ]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Copying"):
            pass

    print("Done.\nTrain dir:", train_audio_dir, "\nVal dir:", val_audio_dir)


if __name__ == "__main__":
    main() 