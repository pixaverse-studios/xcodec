import os
import wget
import tarfile
import json
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

def update_download_progress(data_dir, dataset, total_size, current_size):
    """Update download progress in data_dl.json"""
    progress_file = os.path.join(data_dir, "data_dl.json")
    
    # Load or create progress tracking
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {}
        
    # Update progress for this dataset
    progress[dataset] = {
        "total_size": total_size,
        "downloaded": current_size,
        "percent_complete": round((current_size / total_size) * 100, 2)
    }
    
    # Save updated progress
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def download_librispeech_subset(subset_info):
    subset, url = subset_info
    data_dir = "data"
    
    print(f"\nDownloading {subset}...")
    
    # Download file
    tar_path = os.path.join(data_dir, f"{subset}.tar.gz")
    if not os.path.exists(tar_path):
        wget.download(url, tar_path)
    
    # Extract file
    print(f"\nExtracting {subset}...")
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        total_files = len(members)
        for i, member in enumerate(tqdm(members, desc=f"Extracting {subset}")):
            tar.extract(member, data_dir)
            if i % (total_files // 10) == 0:  # Update every 10%
                update_download_progress(data_dir, "librispeech", total_files, i)
    
    # Remove tar file after extraction
    os.remove(tar_path)
    
    return subset

def download_laion_batch(batch_info):
    start_idx, items, data_dir, split_ratios, total_samples = batch_info
    
    laion_dir = os.path.join(data_dir, "laion")
    audio_dir = os.path.join(laion_dir, "audio")
    
    # Create splits directories
    splits = ["train", "val", "test"]
    split_dirs = {split: os.path.join(laion_dir, split) for split in splits}
    
    results = []
    for i, item in enumerate(items, start=start_idx):
        try:
            # Determine split based on ratios
            rand_val = hash(str(i)) / 2**32
            if rand_val < split_ratios[0]:
                current_split = "train"
            elif rand_val < sum(split_ratios[:2]):
                current_split = "val"
            else:
                current_split = "test"

            # Save audio file
            audio_path = os.path.join(audio_dir, f"laion_{i:09d}.mp3")
            with open(audio_path, "wb") as f:
                f.write(item["audio"]["bytes"])

            # Create entry in split directory
            split_path = os.path.join(split_dirs[current_split], f"laion_{i:09d}.txt")
            with open(split_path, "w") as f:
                f.write(f"{audio_path}\n")
                
            results.append(i)

            # Update progress every 10% of total samples
            if i % (total_samples // 10) == 0:
                update_download_progress(data_dir, "laion", total_samples, i)

        except Exception as e:
            print(f"Error downloading sample {i}: {e}")
            continue
            
    return results

def download_and_extract_datasets():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # LibriSpeech URLs - using full dataset
    librispeech_urls = {
        'train-clean-100': 'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
        'train-clean-360': 'https://www.openslr.org/resources/12/train-clean-360.tar.gz', 
        'train-other-500': 'https://www.openslr.org/resources/12/train-other-500.tar.gz',
        'dev-clean': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
        'dev-other': 'https://www.openslr.org/resources/12/dev-other.tar.gz',
        'test-clean': 'https://www.openslr.org/resources/12/test-clean.tar.gz',
        'test-other': 'https://www.openslr.org/resources/12/test-other.tar.gz'
    }

    # Download and extract LibriSpeech using threads
    print("Processing LibriSpeech dataset...")
    max_workers = min(len(librispeech_urls), cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_librispeech_subset, (subset, url)) 
                  for subset, url in librispeech_urls.items()]
        
        for future in as_completed(futures):
            subset = future.result()
            print(f"Completed processing {subset}")

    print("\nLibriSpeech download and extraction complete!")

    # Download LAION dataset
    print("\nProcessing LAION-Audio dataset...")
    laion_dir = os.path.join(data_dir, "laion")
    os.makedirs(laion_dir, exist_ok=True)
    
    audio_dir = os.path.join(laion_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Create splits directories
    splits = ["train", "val", "test"]
    split_ratios = [0.98, 0.01, 0.01]  # Same as in processor.py
    split_dirs = {split: os.path.join(laion_dir, split) for split in splits}
    for d in split_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Stream and download all LAION samples
    print("Streaming LAION-Audio-300M samples...")
    ds = load_dataset("laion/LAION-Audio-300M", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10_000, seed=42)

    # Process LAION samples in parallel batches
    batch_size = 100
    max_workers = cpu_count()
    total_samples = 300_000_000  # Total samples in LAION-Audio-300M
    
    print("Downloading all LAION samples in parallel (this will take a while)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch = []
        batch_futures = []
        
        for i, item in enumerate(tqdm(ds)):
            batch.append(item)
            
            if len(batch) >= batch_size:
                batch_info = (i - len(batch) + 1, batch, data_dir, split_ratios, total_samples)
                future = executor.submit(download_laion_batch, batch_info)
                batch_futures.append(future)
                batch = []
                
        # Process remaining items
        if batch:
            batch_info = (i - len(batch) + 1, batch, data_dir, split_ratios, total_samples)
            future = executor.submit(download_laion_batch, batch_info)
            batch_futures.append(future)
            
        # Wait for all batches to complete
        for future in as_completed(batch_futures):
            future.result()

    print("\nLAION dataset download complete!")

if __name__ == "__main__":
    download_and_extract_datasets()
