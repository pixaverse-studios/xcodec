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
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
import hydra
import utils
from transformers import AutoFeatureExtractor
from torchaudio.transforms import Resample
from tqdm import tqdm

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ocwd = hydra.utils.get_original_cwd()

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        ds = StandardAudioDataset(phase, self.cfg)
        dl = DataLoader(ds, 
                        batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=16,
                        collate_fn=ds.collate_fn,
                        pin_memory=True,
                        persistent_workers=True)
        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        return self.get_loader('test')

class StandardAudioDataset(Dataset):
    """Dataset for loading standardized audio data
    
    Works with the standardized format created by processor.py
    """
    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        self.ocwd = hydra.utils.get_original_cwd()
        
        # ------------------------------------------------------------------
        # SINGLE- vs MULTI-CORPUS SETUP
        # ------------------------------------------------------------------
        self.multi_corpus = False

        if hasattr(cfg.dataset, 'names') and cfg.dataset.names:
            # New style: list of dataset names to be combined
            self.dataset_names = list(cfg.dataset.names)
            self.dataset_name = self.dataset_names[0]  # primary for logging
            self.multi_corpus = True
        else:
            # Back-compat: single dataset
            if hasattr(cfg.dataset, 'name'):
                self.dataset_name = cfg.dataset.name
            else:
                print("No dataset name specified in config, defaulting to 'librispeech'")
                self.dataset_name = "librispeech"
            self.dataset_names = [self.dataset_name]
        
        # Set data root directory
        if hasattr(cfg.dataset, 'root'):
            self.data_root = join(self.ocwd, cfg.dataset.root)
        else:
            self.data_root = join(self.ocwd, "./data")
            print(f"No data root specified in config, using default: {self.data_root}")
        
        # First check if we have a central registry
        self.registry_path = join(self.data_root, "datasets.json")
        combined_file_list = []
        sr_set = set()

        if exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)

            for dname in self.dataset_names:
                if dname not in registry.get("datasets", {}):
                    raise ValueError(
                        f"Dataset '{dname}' not found in registry. Available: {list(registry.get('datasets', {}).keys())}"
                    )

                dinfo = registry["datasets"][dname]
                droot = join(self.data_root, dinfo["path"])

                tsv_path = join(droot, f"{phase}.tsv")
                if not exists(tsv_path):
                    raise FileNotFoundError(f"Expected TSV {tsv_path} for dataset '{dname}' not found.")

                flist = self.load_tsv(tsv_path)
                # attach root so __getitem__ can build absolute path
                for item in flist:
                    item["_root"] = droot
                combined_file_list.extend(flist)

                sr_set.add(dinfo.get("sample_rate", cfg.preprocess.audio.sr))

            self.file_list = combined_file_list

            # choose sample rate (must match)
            if len(sr_set) > 1:
                raise ValueError(f"Sample rate mismatch across datasets: {sr_set}")
            self.sr = sr_set.pop()
            self.audio_format = "mixed"

            print(f"Loaded {len(self.file_list)} {phase} samples from {len(self.dataset_names)} datasets: {self.dataset_names}")
        else:
            # No registry available – expect a folder at processed/<dataset_name>
            self.dataset_root = join(self.data_root, "processed", self.dataset_name)

            self.metadata_path = join(self.dataset_root, "metadata.json")
            if not exists(self.metadata_path):
                raise FileNotFoundError(
                    f"metadata.json not found for dataset '{self.dataset_name}'. "
                    "Run scripts/processor.py to format & register the dataset."
                )

            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

            self.sr = self.metadata.get("sample_rate", cfg.preprocess.audio.sr)
            self.audio_format = self.metadata.get("audio_format", "flac")
        
        # Load file list from TSV – with auto-discover fallback
        self.tsv_path = join(self.dataset_root, f"{phase}.tsv")

        if not exists(self.tsv_path):
            processed_root = join(self.data_root, "processed")
            if os.path.isdir(processed_root):
                for sub in os.listdir(processed_root):
                    candidate_tsv = join(processed_root, sub, f"{phase}.tsv")
                    if exists(candidate_tsv):
                        print(f"[INFO] Auto-switched dataset_root to '{sub}' (found {phase}.tsv).")
                        self.dataset_root = join(processed_root, sub)
                        self.tsv_path = candidate_tsv
                        break

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
        if "_root" in file_info:
            audio_path = join(file_info["_root"], file_info["path"])
        elif hasattr(self, 'dataset_root'):
            audio_path = join(self.dataset_root, file_info["path"])
        else:
            raise ValueError("Cannot determine audio file path – check dataset registration and TSV paths.")
        
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
