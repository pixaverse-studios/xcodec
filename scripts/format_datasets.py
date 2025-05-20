import os
import shutil
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm
from omegaconf import OmegaConf

REGISTRY_NAME = "datasets.json"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def scan_audio(root: Path, include_exts=(".wav", ".flac", ".mp3")) -> List[Path]:
    """Recursively find audio files under *root* with given extensions."""
    files: List[Path] = []
    for ext in include_exts:
        files.extend(root.rglob(f"*{ext}"))
    return files


def write_tsv(file_infos: List[Dict], out_path: Path):
    with out_path.open("w") as f:
        for fi in file_infos:
            f.write(f"{fi['path']}\t{fi['duration']}\n")
    print(f"✓ Wrote {len(file_infos)} lines → {out_path}")


# ------------------------------------------------------------------
# Core
# ------------------------------------------------------------------

def format_single_dataset(d_cfg, registry: Dict, output_base: Path):
    """Format one dataset according to config block *d_cfg*"""
    
    print(f"Processing dataset config: {d_cfg}")  # Debug log
    
    name = d_cfg["name"]  # Access as dict instead of attribute
    src_dir: Path = Path(d_cfg["src_dir"]).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir {src_dir} not found for dataset '{name}'")

    # --------------------------------------------------------------
    # Output skeleton
    # --------------------------------------------------------------
    ds_out_dir = output_base / "processed" / name
    audio_out_dir = ds_out_dir / "audio"
    ds_out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Scan + split
    # --------------------------------------------------------------
    files = scan_audio(src_dir)
    random.shuffle(files)
    total = len(files)
    if total == 0:
        raise RuntimeError(f"No audio files found under {src_dir}")
    print(f"Dataset '{name}': found {total} audio files.")

    split_ratio = d_cfg.get("split", {"train": 0.9, "val": 0.1})
    train_cut = int(total * split_ratio.get("train", 0.9))
    val_cut = train_cut + int(total * split_ratio.get("val", 0.1))

    split_map = {}
    for idx, p in enumerate(files):
        if idx < train_cut:
            split_map[p] = "train"
        elif idx < val_cut:
            split_map[p] = "val"
        else:
            split_map[p] = "test"

    # --------------------------------------------------------------
    # Copy and collect metadata
    # --------------------------------------------------------------
    processed_files: List[Dict] = []
    sample_rate = 16000  # default; training code will resample

    for src_path in tqdm(files, desc=f"Copying→{name}"):
        try:
            duration = -1  # unknown; can be filled later if needed

            tag = d_cfg.get("tag", name)
            new_name = f"{tag}_{src_path.stem}{src_path.suffix}"
            dst_path = audio_out_dir / new_name
            safe_copy(src_path, dst_path)

            processed_files.append(
                {
                    "original_path": str(src_path),
                    "path": str(Path("audio") / new_name),
                    "duration": duration,
                    "split": split_map[src_path],
                }
            )
        except Exception as e:
            print(f"[WARN] Failed to process {src_path}: {e}")

    # --------------------------------------------------------------
    # Write metadata & TSVs
    # --------------------------------------------------------------
    metadata = {
        "dataset": name,
        "splits": {s: s for s in ["train", "val", "test"]},
        "sample_rate": sample_rate,
        "audio_format": "mixed",
        "files": processed_files,
    }

    with (ds_out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    splits = {"train": [], "val": [], "test": []}
    for fi in processed_files:
        splits[fi["split"]].append(fi)

    for split, lst in splits.items():
        if lst:
            write_tsv(lst, ds_out_dir / f"{split}.tsv")

    # --------------------------------------------------------------
    # Update registry dict in-place
    # --------------------------------------------------------------
    registry.setdefault("datasets", {})[name] = {
        "path": f"processed/{name}",
        "name": name,
        "sample_rate": metadata["sample_rate"],
        "audio_format": metadata["audio_format"],
        "splits": metadata["splits"],
        "last_updated": None,
        "stats": {
            "train_samples": len(splits["train"]),
            "val_samples": len(splits["val"]),
            "test_samples": len(splits["test"]),
        },
    }

    print(f"✓ Dataset '{name}' formatted. Output in {ds_out_dir}\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Format datasets into the standard layout")
    parser.add_argument("config", nargs="?", default=None, help="Optional YAML/JSON config. If omitted, auto-discovers datasets under ./extracted.")
    parser.add_argument("--extracted_root", type=str, default="./extracted", help="Directory containing extracted dataset folders (used when no config).")
    parser.add_argument("--output_root", type=str, default="./data", help="Root where 'processed/' and registry live")
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Build a python list `datasets_list` from whichever config style is used
    # --------------------------------------------------------------

    datasets_list = []

    if args.config is not None:
        print(f"[DEBUG] Using config file: {args.config}")
        cfg = OmegaConf.load(args.config)
        print(f"Loaded config: {OmegaConf.to_container(cfg)}")  # Debug log

        if "datasets" in cfg:  # explicit list style
            for ds in cfg.datasets:
                datasets_list.append(OmegaConf.to_container(ds))  # Convert to dict
        elif "dataset" in cfg and "names" in cfg.dataset:
            # Optional explicit src_root, otherwise fall back to --extracted_root
            src_root = Path(cfg.dataset.get("src_root", args.extracted_root))
            for n in cfg.dataset.names:
                datasets_list.append({"name": n, "src_dir": str(src_root / n), "tag": n})
        else:
            raise ValueError("Config missing 'datasets' list or 'dataset.names'.")
    else:
        # No config: auto-discover under extracted_root
        extracted_root = Path(args.extracted_root).resolve()
        print(f"[DEBUG] Auto-discovering datasets under {extracted_root}")
        if not extracted_root.exists():
            raise FileNotFoundError(f"Extracted root {extracted_root} not found. Run extract_archives.py first.")

        for sub in extracted_root.iterdir():
            if sub.is_dir():
                datasets_list.append({"name": sub.name, "src_dir": str(sub), "tag": sub.name})

    print(f"Datasets to process: {datasets_list}")  # Debug log

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    registry_path = output_root / REGISTRY_NAME
    registry = {"datasets": {}}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())

    for ds_cfg in datasets_list:
        format_single_dataset(ds_cfg, registry, output_root)

    # save registry
    with registry_path.open("w") as f:
        json.dump(registry, f, indent=2)
    print(f"Updated registry → {registry_path}")


if __name__ == "__main__":
    main() 