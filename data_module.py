import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
import random
import librosa
import json
import hydra
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
import utils
from transformers import AutoFeatureExtractor
from torchaudio.transforms import Resample
from tqdm import tqdm

# new for CombinedAudioDataset
from itertools import accumulate
import bisect

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ocwd = hydra.utils.get_original_cwd()

    # --------------------------------------------------------------
    # internal
    # --------------------------------------------------------------

    def _build_dataset(self, phase):
        # If config supplies a *names* list => combined dataset
        if hasattr(self.cfg.dataset, 'names'):
            return CombinedAudioDataset(phase, self.cfg)
        else:
            return StandardAudioDataset(phase, self.cfg)

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size

        ds = self._build_dataset(phase)

        dl = DataLoader(
            ds,
                        batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=16,
                        collate_fn=ds.collate_fn,
                        pin_memory=True,
            persistent_workers=True,
        )
        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        return self.get_loader('test')

class StandardAudioDataset(Dataset):
    """Dataset for loading a single standardized dataset (LibriSpeech, LAION, etc.).
    
    A *dataset_name* override is accepted so that multiple datasets can be
    combined without having to clone the whole Hydra config object.
    """
    def __init__(self, phase, cfg, dataset_name: str | None = None):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        self.ocwd = hydra.utils.get_original_cwd()
        
        # Get dataset config - default to "librispeech" if not specified
        # This handles older config files that don't have dataset.name
        if dataset_name is not None:
            self.dataset_name = dataset_name
        elif hasattr(cfg.dataset, 'name'):
            self.dataset_name = cfg.dataset.name
        else:
            print("No dataset name specified in config, defaulting to 'librispeech'")
            self.dataset_name = "librispeech"
        
        # Set data root directory
        if hasattr(cfg.dataset, 'root'):
            self.data_root = join(self.ocwd, cfg.dataset.root)
        else:
            self.data_root = join(self.ocwd, "./data")
            print(f"No data root specified in config, using default: {self.data_root}")
        
        # First check if we have a central registry
        self.registry_path = join(self.data_root, "datasets.json")
        if exists(self.registry_path):
            # Use the central registry to get dataset information
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
                
            if self.dataset_name not in registry.get("datasets", {}):
                raise ValueError(f"Dataset '{self.dataset_name}' not found in registry. Available datasets: {list(registry.get('datasets', {}).keys())}")
                
            dataset_info = registry["datasets"][self.dataset_name]
            self.dataset_root = join(self.data_root, dataset_info["path"])
            
            # Load dataset-specific parameters from registry
            self.sr = dataset_info.get("sample_rate", cfg.preprocess.audio.sr)
            self.audio_format = dataset_info.get("audio_format", "flac")
            
            print(f"Using dataset '{self.dataset_name}' from registry")
            print(f"Stats: Train: {dataset_info['stats'].get('train_samples', 0)} samples, "
                  f"Val: {dataset_info['stats'].get('val_samples', 0)} samples, "
                  f"Test: {dataset_info['stats'].get('test_samples', 0)} samples")
        else:
            # Fallback to direct folder structure if no registry
            self.dataset_root = join(self.data_root, "processed", self.dataset_name)
            
            # Load metadata from dataset folder
            self.metadata_path = join(self.dataset_root, "metadata.json")
            if exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.sr = self.metadata.get("sample_rate", cfg.preprocess.audio.sr)
                self.audio_format = self.metadata.get("audio_format", "flac")
            else:
                # Last resort: try to use LibriSpeech paths directly from config
                if hasattr(cfg.preprocess, 'datasets') and hasattr(cfg.preprocess.datasets, 'LibriSpeech'):
                    print(f"No metadata found for dataset '{self.dataset_name}', using LibriSpeech config")
                    
                    # Use TSV files from the config
                    if self.phase == 'train':
                        self.tsv_path = join(self.ocwd, cfg.preprocess.view.train_filelist)
                    elif self.phase == 'val':
                        self.tsv_path = join(self.ocwd, cfg.dataset.val.filelist)
                    else:
                        self.tsv_path = join(self.ocwd, cfg.preprocess.view.test_filelist)
                    
                    self.sr = cfg.preprocess.audio.sr
                    self.audio_format = "flac"
                    self.file_list = self.load_tsv(self.tsv_path)
                    self.min_audio_length = cfg.dataset.min_audio_length
                    self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
                    
                    print(f"Loaded {len(self.file_list)} {phase} samples from {self.tsv_path}")
                    return
                else:
                    raise FileNotFoundError(f"Neither registry nor metadata found for dataset '{self.dataset_name}'")
        
        # Load file list from TSV
        self.tsv_path = join(self.dataset_root, f"{phase}.tsv")
        self.file_list = self.load_tsv(self.tsv_path)
        
        # Set minimum audio length
        self.min_audio_length = cfg.dataset.min_audio_length
        
        # Initialize feature extractor for audio representation
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        
        print(f"Loaded {len(self.file_list)} {phase} samples from {self.dataset_name}")
        
    def __len__(self):
        return len(self.file_list)

    def load_tsv(self, tsv_path):
        """Load file list from TSV file"""
        if not exists(tsv_path):
            raise FileNotFoundError(f"TSV file not found at {tsv_path}")
            
        file_list = []
        with open(tsv_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    path = parts[0]
                    duration = int(parts[1])
                    file_list.append({"path": path, "duration": duration})
        return file_list

    def __getitem__(self, idx):
        file_info = self.file_list[idx]
        
        # Handle different path formats - for compatibility with old config
        if hasattr(self, 'dataset_root'):
            audio_path = join(self.dataset_root, file_info["path"])
        elif hasattr(self.cfg.preprocess.datasets, 'LibriSpeech'):
            # Direct LibriSpeech path from old config
            audio_path = join(self.cfg.preprocess.datasets.LibriSpeech.root, file_info["path"])
        else:
            raise ValueError("Cannot determine audio file path")
        
        # Verify file exists
        if not exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")
        
        # Load audio file
        try:
            wav, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != 16000:
                wav = Resample(sr, 16000)(wav)
                
            # Take first channel if multichannel
            wav = wav[0, :]
            
            # Handle audio length
            length = wav.shape[0]
            
            # Pad if too short
            if length < self.min_audio_length:
                wav = F.pad(wav, (0, self.min_audio_length - length))
                length = wav.shape[0]
                
            # Random crop if too long
            if length > self.min_audio_length:
                i = random.randint(0, length - self.min_audio_length)
                wav = wav[i:i + self.min_audio_length]
            
            # Add padding for the feature extractor
            wav_pad = F.pad(wav, (160, 160))
            
            # Extract features
            feat = self.feature_extractor(wav_pad, sampling_rate=16000, return_tensors="pt").data['input_features']
            
            # Prepare output
            out = {
                'wav': wav,
                'feat': feat,
                'path': audio_path
            }
            
            return out
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {str(e)}")
    
    def collate_fn(self, batch):
        """Collate function for batching samples"""
        wavs = [b['wav'] for b in batch]
        wavs = torch.stack(wavs)
        
        feats = [b['feat'] for b in batch]
        feats = torch.stack(feats)
        
        paths = [b['path'] for b in batch]
        
        out = {
            'wav': wavs,  
            'feats': feats,
            'paths': paths
        }
        return out

# -----------------------------------------------------------------------------
# Support for training on MULTIPLE datasets simultaneously
# -----------------------------------------------------------------------------

class CombinedAudioDataset(Dataset):
    """Concatenate several StandardAudioDataset instances.

    Example in config:

    dataset:
      names: [librispeech, laion]

    This class keeps each sub-dataset's own TSV list but exposes a flat index-space.
    """

    def __init__(self, phase: str, cfg):
        from copy import deepcopy
        from itertools import accumulate

        self.subsets = []
        self.cum_lens = []

        dataset_names = list(cfg.dataset.names)

        total = 0
        for name in dataset_names:
            cfg_single = deepcopy(cfg)
            cfg_single.dataset.name = name
            subset = StandardAudioDataset(phase, cfg_single, dataset_name=name)
            self.subsets.append(subset)
            total += len(subset)
            self.cum_lens.append(total)

        # Re-use collate_fn from first subset (identical across all)
        self._collate_fn = self.subsets[0].collate_fn if self.subsets else lambda x: x

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self):
        return self.cum_lens[-1] if self.cum_lens else 0

    def __getitem__(self, idx):
        # binary search for subset index
        import bisect
        subset_idx = bisect.bisect_right(self.cum_lens, idx)
        previous_cum = 0 if subset_idx == 0 else self.cum_lens[subset_idx - 1]
        local_idx = idx - previous_cum
        return self.subsets[subset_idx][local_idx]

    # expose collate_fn so DataLoader can use it
    def collate_fn(self, batch):
        return self._collate_fn(batch)

@hydra.main(config_path='config', config_name='default')
def main(cfg):
    """Main function for testing the dataset"""
    data_module = DataModule(cfg)
    
    # Test the validation dataloader
    val_loader = data_module.val_dataloader()
    
    # Process a few batches to test
    valid_paths = []
    max_batches = 3
    
    print("Testing dataloader with a few batches:")
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Testing batches", unit="batch")):
        if batch_idx >= max_batches:
            break
            
        wavs = batch['wav']
        paths = batch['paths']
        print(f"Batch {batch_idx+1}: Shape={wavs.shape}, Min={wavs.min().item()}, Max={wavs.max().item()}")
        valid_paths.extend(paths)
    
    print(f"Successfully loaded {len(valid_paths)} audio files")
    print("Random sample paths:")
    for path in random.sample(valid_paths, min(5, len(valid_paths))):
        print(f" - {path}")

if __name__ == "__main__":
    main()

