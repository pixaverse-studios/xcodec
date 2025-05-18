import os
import wget
import tarfile
import concurrent.futures
import requests
from tqdm import tqdm
from datasets import load_dataset

def download_file(url, path):
    """Download a file with progress bar and better chunk size"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # Increased chunk size for better performance

    with open(path, 'wb') as f, tqdm(
        desc=os.path.basename(path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def download_laion_sample(args):
    """Helper function to download a single LAION sample"""
    i, item, audio_dir, split_dirs, split_ratios = args
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
        
        return True
    except Exception as e:
        print(f"Error downloading sample {i}: {e}")
        return False

def download_and_extract_datasets():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # LibriSpeech URL - using only train-other-500 (largest subset)
    librispeech_url = 'https://www.openslr.org/resources/12/train-other-500.tar.gz'

    # Download and extract LibriSpeech
    print("Processing LibriSpeech dataset...")
    print("\nDownloading train-other-500...")
    
    # Download file with better performance
    tar_path = os.path.join(data_dir, "train-other-500.tar.gz")
    if not os.path.exists(tar_path):
        download_file(librispeech_url, tar_path)
    
    # Extract file using larger buffer size
    print(f"\nExtracting train-other-500...")
    with tarfile.open(tar_path, bufsize=256*1024) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting train-other-500"):
            tar.extract(member, data_dir)
    
    # Remove tar file after extraction
    os.remove(tar_path)

    print("\nLibriSpeech download and extraction complete!")

    # Download LAION dataset
    print("\nProcessing LAION-Audio dataset...")
    laion_dir = os.path.join(data_dir, "laion")
    os.makedirs(laion_dir, exist_ok=True)

    # Stream and download all LAION samples
    print("Streaming LAION-Audio-300M samples...")
    ds = load_dataset("laion/LAION-Audio-300M", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10_000, seed=42)

    audio_dir = os.path.join(laion_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Create splits directories
    splits = ["train", "val", "test"]
    split_ratios = [0.98, 0.01, 0.01]  # Same as in processor.py
    split_dirs = {split: os.path.join(laion_dir, split) for split in splits}
    for d in split_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Use ThreadPoolExecutor for parallel downloads
    print("Downloading LAION samples in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for i, item in enumerate(ds):
            args = (i, item, audio_dir, split_dirs, split_ratios)
            futures.append(executor.submit(download_laion_sample, args))

            # Process in batches to avoid memory issues
            if len(futures) >= 100:
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    _ = future.result()
                futures = []

        # Process remaining futures
        if futures:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                _ = future.result()

    print("\nLAION dataset download complete!")

if __name__ == "__main__":
    download_and_extract_datasets()
