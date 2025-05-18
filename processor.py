import os
import argparse
import torchaudio
import torch
import json
import shutil
import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Standard format for all datasets:
# data/
#   datasets.json  # Central registry of all processed datasets
#   processed/
#     dataset_name/
#       metadata.json  # Dataset-specific metadata
#       audio/  # All audio files with standardized names
#       train.tsv  # Training split with standardized paths
#       val.tsv    # Validation split
#       test.tsv   # Test split

def update_dataset_registry(output_base_dir, dataset_name, metadata):
    """Update the central dataset registry file"""
    registry_path = os.path.join(output_base_dir, "datasets.json")
    
    # Create or load existing registry
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"datasets": {}}
    
    # Add or update dataset information
    registry["datasets"][dataset_name] = {
        "path": f"processed/{dataset_name}",
        "name": metadata["dataset"],
        "sample_rate": metadata["sample_rate"],
        "audio_format": metadata["audio_format"],
        "splits": metadata["splits"],
        "last_updated": datetime.datetime.now().isoformat(),
        "stats": {
            "train_samples": len([f for f in metadata.get("files", []) if f.get("split") == "train"]),
            "val_samples": len([f for f in metadata.get("files", []) if f.get("split") == "val"]),
            "test_samples": len([f for f in metadata.get("files", []) if f.get("split") == "test"])
        }
    }
    
    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"Updated dataset registry at {registry_path}")

def standardize_audio(input_path, output_path, target_sr=16000):
    """Standardize audio to consistent format"""
    try:
        # Load audio
        waveform, sr = torchaudio.load(input_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
            
        # Save standardized audio
        torchaudio.save(output_path, waveform, target_sr)
        
        return waveform.shape[1]  # Return number of samples (as int)
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None

def process_dataset(input_dir, output_dir, dataset_name, split_name, output_base_dir):
    """Generic dataset processor that standardizes any audio dataset"""
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    metadata = {
        "dataset": dataset_name,
        "splits": {},
        "sample_rate": 16000, # Standard sample rate
        "audio_format": "wav",
        "files": []
    }
    
    processed_files = []
    
    # Walk through all audio files recursively
    for root, _, files in tqdm(os.walk(input_dir)):
        for audio_file in files:
            if audio_file.endswith(('.wav', '.mp3', '.flac')):
                
                # Create standardized filename
                file_id = len(processed_files)
                std_name = f"{dataset_name}_{file_id:09d}.wav"
                
                src_path = os.path.join(root, audio_file)
                dst_path = os.path.join(audio_dir, std_name)

                # Standardize audio format and get duration
                duration = standardize_audio(src_path, dst_path)
                if duration is None:
                    continue
                    
                # Determine split (can be customized per dataset)
                if "train" in root.lower():
                    split = "train"
                elif "val" in root.lower() or "dev" in root.lower():
                    split = "val"
                elif "test" in root.lower():
                    split = "test"
                else:
                    split = "train" # Default to train

                        file_info = {
                    "original_path": os.path.relpath(src_path, input_dir),
                            "path": os.path.join("audio", std_name),
                            "duration": duration,
                    "split": split
                        }
                        
                        processed_files.append(file_info)
                        metadata["files"].append(file_info)
                metadata["splits"][split] = split
                        
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update registry
    update_dataset_registry(output_base_dir, dataset_name, metadata)
    
    # Create TSV files
    splits = {"train": [], "val": [], "test": []}
    for file_info in processed_files:
        splits[file_info["split"]].append(file_info)
    
    for split, files in splits.items():
        if not files:
            continue
        with open(os.path.join(output_dir, f"{split}.tsv"), 'w') as f:
            for fi in files:
                f.write(f"{fi['path']}\t{fi['duration']}\n")
                
    print(f"Processed {len(processed_files)} files from {dataset_name}")
    return output_dir

def process_librispeech(input_dir, output_dir, split_name, output_base_dir):
    """Process LibriSpeech dataset into standardized format"""
    return process_dataset(input_dir, output_dir, "librispeech", split_name, output_base_dir)

def process_voxceleb(input_dir, output_dir, split_name, output_base_dir):
    """Process VoxCeleb dataset into standardized format"""
    return process_dataset(input_dir, output_dir, "voxceleb", split_name, output_base_dir)

def process_mls(input_dir, output_dir, split_name, output_base_dir):
    """Process MLS dataset into standardized format"""
    return process_dataset(input_dir, output_dir, "mls", split_name, output_base_dir)

def process_custom(input_dir, output_dir, split_name, output_base_dir):
    """Process custom dataset into standardized format"""
    return process_dataset(input_dir, output_dir, "custom", split_name, output_base_dir)

def main():
    parser = argparse.ArgumentParser(description="Process audio datasets into a standardized format")
    parser.add_argument("--dataset", type=str, required=True, choices=["librispeech", "laion", "voxceleb", "mls", "custom"],
                        help="Dataset type to process")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the original dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--split", type=str, default="all", help="Split to process (train, val, test, or all)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_subdir = os.path.join(args.output_dir, "processed", args.dataset)
    
    # Process dataset based on type
    if args.dataset == "librispeech":
        process_librispeech(args.input_dir, output_subdir, args.split, args.output_dir)
    elif args.dataset == "laion":
        process_dataset(args.input_dir, output_subdir, "laion", args.split, args.output_dir)
    elif args.dataset == "voxceleb":
        process_voxceleb(args.input_dir, output_subdir, args.split, args.output_dir)
    elif args.dataset == "mls":
        process_mls(args.input_dir, output_subdir, args.split, args.output_dir)
    elif args.dataset == "custom":
        process_custom(args.input_dir, output_subdir, args.split, args.output_dir)
    
    print(f"Dataset processed and saved to: {output_subdir}")
    print(f"Use this dataset by setting 'name: {args.dataset}' in your config file")

if __name__ == "__main__":
    main() 