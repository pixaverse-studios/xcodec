import os
import shutil
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

import torchaudio
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

    name: str = d_cfg.name  # required
    src_dir: Path = Path(d_cfg.src_dir).resolve()
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
    sample_rate = None

    for src_path in tqdm(files, desc=f"Copying→{name}"):
        try:
            info = torchaudio.info(str(src_path))
            duration = info.num_frames
            if sample_rate is None:
                sample_rate = info.sample_rate
        except Exception:
            # fall back to duration unknown
            duration = -1

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

    # --------------------------------------------------------------
    # Write metadata & TSVs
    # --------------------------------------------------------------
    metadata = {
        "dataset": name,
        "splits": {s: s for s in ["train", "val", "test"]},
        "sample_rate": sample_rate or 16000,
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
    parser = argparse.ArgumentParser(description="Format one or more datasets into the standard layout")
    parser.add_argument("config", type=str, help="YAML/JSON config describing datasets to format")
    parser.add_argument("--output_root", type=str, default="./data", help="Root where 'processed/' and registry live")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # ------------------------------------------------------------------
    # Accept two config styles:
    #   1) Explicit list – cfg.datasets: [ {name, src_dir, ...}, ... ]
    #   2) Hydra-style dataset block – cfg.dataset.names + cfg.dataset.root
    # ------------------------------------------------------------------

    if not hasattr(cfg, "datasets"):
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "names"):
            root_dir = Path(cfg.dataset.get("root", "./data"))
            ds_entries = []
            for n in cfg.dataset.names:
                ds_entries.append({"name": n, "src_dir": str(Path(root_dir) / n), "tag": n})
            cfg.datasets = ds_entries  # type: ignore[attr-defined]
        else:
            raise ValueError("Config must contain either 'datasets' list or 'dataset.names'.")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    registry_path = output_root / REGISTRY_NAME
    registry = {"datasets": {}}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())

    for ds_cfg in cfg.datasets:
        format_single_dataset(ds_cfg, registry, output_root)

    # save registry
    with registry_path.open("w") as f:
        json.dump(registry, f, indent=2)
    print(f"Updated registry → {registry_path}")


if __name__ == "__main__":
    main() 