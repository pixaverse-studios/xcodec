import os
import random
import json
from os.path import join, exists
from typing import Dict

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, IterableDataset

import hydra
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoFeatureExtractor
from torchaudio.transforms import Resample

# Re-use the original non-streaming dataset implementation
from data_module import StandardAudioDataset  # noqa: E402

__all__ = ["DataModule"]

# ------------------------------------------------------------------
# HARD-CODED SWITCH – set to True to *always* stream from HuggingFace.
# Ignores whatever is in the Hydra config.
# ------------------------------------------------------------------
STREAMING_MODE = False

class HFAudioStreamingDataset(IterableDataset):
    """Stream audio examples from 🤗 datasets.

    Expects cfg.dataset to provide:
        hf_name:   dataset name on the Hub (e.g. "librispeech_asr")
        hf_config: optional config (language subset etc.)
        hf_splits: mapping {train|val|test: split_name}
    """

    def __init__(self, phase: str, cfg):
        self.phase = phase
        self.cfg = cfg

        d_cfg = cfg.dataset
        # Use .get to avoid Hydra struct errors if key missing
        hf_name = d_cfg.get("hf_name", "laion/LAION-Audio-300M")
        hf_config = d_cfg.get("hf_config", None)
        split_map: Dict[str, str] = d_cfg.get(
            "hf_splits", {"train": "train", "val": "train", "test": "train"}
        )
        split = split_map.get(phase, "train")

        self.raw_ds = load_dataset(hf_name, hf_config, split=split, streaming=True)

        # Optional: limit the number of streamed examples
        max_n = d_cfg.get("max_stream_samples", 10000 )
        print(f"max_n: {max_n}")
        self._max_n = int(max_n) if max_n is not None else None
        if self._max_n is not None:
            self.raw_ds = self.raw_ds.take(self._max_n)

        # Optional shuffling (only meaningful during training)
        if phase == "train" and d_cfg.get("shuffle", True):
            buffer_size = int(d_cfg.get("shuffle_buffer_size", 10_000))
            self.raw_ds = self.raw_ds.shuffle(buffer_size=buffer_size, seed=42)

        self.min_audio_length = d_cfg.get("min_audio_length", 96000)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------
    def __iter__(self):
        for item in self.raw_ds:
            example = self._process_item(item)
            if example is not None:
                yield example

    # If a maximum sample count is known, provide __len__ so progress bars show ETA
    def __len__(self):
        if self._max_n is not None:
            return self._max_n
        raise TypeError("Length not defined for unlimited streaming dataset")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _process_item(self, item):
        audio_field = (
            item.get("audio")
            or item.get("audio.mp3")
            or item.get("audio.flac")
            or item.get("audio.wav")
        )
        if audio_field is None:
            return None

        wav = torch.from_numpy(audio_field["array"]).float()
        sr = audio_field["sampling_rate"]

        # mono
        if wav.ndim > 1:
            wav = wav[0]
        if sr != 16000:
            wav = Resample(sr, 16000)(wav)

        # crop / pad
        L = wav.shape[0]
        tgt = self.min_audio_length
        if L < tgt:
            wav = F.pad(wav, (0, tgt - L))
        elif L > tgt:
            i = random.randint(0, L - tgt)
            wav = wav[i : i + tgt]

        wav_pad = F.pad(wav, (160, 160))
        feat = self.feature_extractor(
            wav_pad, sampling_rate=16000, return_tensors="pt"
        ).data["input_features"]

        return {
            "wav": wav,
            "feat": feat,
            "path": item.get("file", "") or item.get("id", "")
        }

    # For DataLoader
    def collate_fn(self, batch):
        wavs = torch.stack([b["wav"] for b in batch])
        feats = torch.stack([b["feat"] for b in batch])
        paths = [b["path"] for b in batch]
        return {"wav": wavs, "feats": feats, "paths": paths}


class DataModule(pl.LightningDataModule):
    """Hybrid DataModule – streams from HF or reads local TS V based on cfg."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    # ------------------------------------------------------------------
    def _select_dataset(self, phase):
        if STREAMING_MODE:
            return HFAudioStreamingDataset(phase, self.cfg)
        # Fall back to original local dataset
        return StandardAudioDataset(phase, self.cfg)

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size

        ds = self._select_dataset(phase)
        # IterableDataset cannot be shuffled by DataLoader
        shuffle_flag = phase_cfg.shuffle and not isinstance(ds, IterableDataset)

        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=phase_cfg.get("num_workers", 8),
            collate_fn=ds.collate_fn,
            pin_memory=True,
        )

        # ------------------------------------------------------------------
        # Lightning cannot compute len(dataloader) for an IterableDataset.
        # If we *do* know the max number of samples, monkey-patch __len__ so
        # progress bars show proper totals and ETA.
        # ------------------------------------------------------------------
        if isinstance(ds, IterableDataset) and hasattr(ds, "_max_n") and ds._max_n is not None:
            total_batches = (ds._max_n + batch_size - 1) // batch_size

            def _patched_len():  # noqa: ANN001
                return total_batches

            dl.__len__ = _patched_len  # type: ignore[attr-defined]

        return dl

    # Lightning hooks
    def train_dataloader(self):
        return self.get_loader("train")

    def val_dataloader(self):
        return self.get_loader("val")

    def test_dataloader(self):
        return self.get_loader("test")


# For quick standalone test
if __name__ == "__main__":
    import omegaconf, yaml

    # Minimal config stub
    cfg_yaml = """
    dataset:
      streaming: true
      hf_name: librispeech_asr
      hf_splits:
        train: train.clean.100
        val: validation.clean
        test: test.clean
      batch_size: 2
      shuffle: true
      shuffle_buffer_size: 1000
      min_audio_length: 96000
      max_stream_samples: 500000
    preprocess:
      audio:
        sr: 16000
    """
    cfg = omegaconf.OmegaConf.create(yaml.safe_load(cfg_yaml))
    dm = DataModule(cfg)
    for batch in dm.train_dataloader():
        print(batch["wav"].shape, batch["feats"].shape)
        break 