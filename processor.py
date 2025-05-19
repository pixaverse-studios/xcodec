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

# Standard format:
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

def process_librispeech(input_dir, output_dir, split_name, output_base_dir):
    """Process LibriSpeech dataset into standardized format"""
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    # Map LibriSpeech splits to our standard splits
    split_mapping = {
        "train-clean-100": "train",
        "train-clean-360": "train",
        "train-other-500": "train",
        "dev-clean": "val",
        "dev-other": "val",
        "test-clean": "test",
        "test-other": "test"
    }
    
    # Create metadata.json
    metadata = {
        "dataset": "LibriSpeech",
        "splits": {},
        "sample_rate": 16000,
        "audio_format": "flac",
        "files": []  # Will store all processed file info
    }
    
    processed_files = []
    
    # Process each LibriSpeech split
    for split in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(input_dir, split)):
            continue
            
        if split not in split_mapping:
            print(f"Skipping unknown split: {split}")
            continue
            
        standard_split = split_mapping[split]
        metadata["splits"][split] = standard_split
        
        split_dir = os.path.join(input_dir, split)
        print(f"Processing {split} -> {standard_split}...")
        
        # Walk through all audio files
        for speaker_id in tqdm(os.listdir(split_dir)):
            speaker_dir = os.path.join(split_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
                
            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                if not os.path.isdir(chapter_dir):
                    continue
                    
                for audio_file in os.listdir(chapter_dir):
                    if not audio_file.endswith('.flac'):
                        continue
                        
                    # Parse LibriSpeech filename format: {speaker_id}-{chapter_id}-{utterance_id}.flac
                    base_name = os.path.splitext(audio_file)[0]
                    utterance_id = base_name.split('-')[-1]
                    
                    # Create standardized filename
                    std_name = f"librispeech_{speaker_id}_{chapter_id}_{utterance_id}.flac"
                    src_path = os.path.join(chapter_dir, audio_file)
                    dst_path = os.path.join(audio_dir, std_name)
                    
                    # Process audio: copy file and get duration
                    try:
                        waveform, sample_rate = torchaudio.load(src_path)
                        duration = waveform.shape[1]
                        
                        # Copy audio file to standardized location
                        shutil.copy2(src_path, dst_path)
                        
                        # File info dictionary
                        file_info = {
                            "original_path": os.path.join(split, speaker_id, chapter_id, audio_file),
                            "path": os.path.join("audio", std_name),
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                            "utterance_id": utterance_id,
                            "duration": duration,
                            "split": standard_split
                        }
                        
                        # Add to processed files
                        processed_files.append(file_info)
                        
                        # Also add to metadata
                        metadata["files"].append(file_info)
                        
                    except Exception as e:
                        print(f"Error processing {src_path}: {e}")
    
    # Save dataset-specific metadata.json
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update central registry
    update_dataset_registry(output_base_dir, "librispeech", metadata)
    
    # Create TSV files for each split
    splits = {"train": [], "val": [], "test": []}
    for file_info in processed_files:
        splits[file_info["split"]].append(file_info)
    
    for split, files in splits.items():
        if not files:
            continue
            
        with open(os.path.join(output_dir, f"{split}.tsv"), 'w') as f:
            for file_info in files:
                f.write(f"{file_info['path']}\t{file_info['duration']}\n")
                
        print(f"Created {split}.tsv with {len(files)} entries")
    
    print(f"Processed {len(processed_files)} files from LibriSpeech")
    return output_dir

def process_mls(input_dir, output_dir, split_name, output_base_dir):
    """Process MLS (Multilingual LibriSpeech) dataset into standardized format"""
    # Implementation for MLS dataset processing
    pass

def process_voxceleb(input_dir, output_dir, split_name, output_base_dir):
    """Process VoxCeleb dataset into standardized format"""
    # Implementation for VoxCeleb dataset processing
    pass

def process_custom(input_dir, output_dir, split_name, output_base_dir):
    """Process a custom dataset with a different structure"""
    # Implementation for custom dataset processing
    pass

def main():
    parser = argparse.ArgumentParser(description="Process audio datasets into a standardized format")
    parser.add_argument("--dataset", type=str, required=True, choices=["librispeech", "mls", "voxceleb", "custom"],
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
    elif args.dataset == "mls":
        process_mls(args.input_dir, output_subdir, args.split, args.output_dir)
    elif args.dataset == "voxceleb":
        process_voxceleb(args.input_dir, output_subdir, args.split, args.output_dir)
    elif args.dataset == "custom":
        process_custom(args.input_dir, output_subdir, args.split, args.output_dir)
    
    print(f"Dataset processed and saved to: {output_subdir}")
    print(f"Use this dataset by setting 'name: {args.dataset}' in your config file")

if __name__ == "__main__":
    main() 