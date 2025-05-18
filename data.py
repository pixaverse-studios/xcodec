import os
import wget
import tarfile
import concurrent.futures
import requests
import shutil
from tqdm import tqdm
from datasets import load_dataset, Audio
import torchaudio, torch

def download_file(url, path):
    """Download a file with progress bar and better chunk size"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024*1024  # Increased to 1MB chunks for better throughput

    # Set TCP keepalive and larger receive buffer
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=100,
        pool_maxsize=100,
        max_retries=3,
        pool_block=False
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    response = session.get(url, stream=True)

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

def save_laion_item_locally(item_dict, index, audio_dir_base, split_dirs_base, split_ratios):
    """
    Processes a single item from the LAION dataset.
    - Determines train/val/test split.
    - Saves the audio array directly as a WAV file.
    - Creates a .txt manifest entry.
    """
    try:
        current_split = "train" # Default
        rand_val = hash(str(index)) / 2**32 # Use index for deterministic split
        if rand_val < split_ratios[0]:
            current_split = "train"
        elif rand_val < sum(split_ratios[:2]):
            current_split = "val"
        else:
            current_split = "test"

        audio_feature = item_dict.get("audio.mp3")

        if audio_feature is None or not isinstance(audio_feature, dict) or \
           "array" not in audio_feature or "sampling_rate" not in audio_feature:
            print(f"Skipping sample {index}: 'audio.mp3' field missing, malformed, or lacks 'array'/'sampling_rate'. Keys: {list(item_dict.keys())}")
            if isinstance(audio_feature, dict):
                 print(f"Audio feature keys: {list(audio_feature.keys())}")
            return None # Indicate failure to process

        audio_filename = f"laion_{index:09d}.wav" # Always save as WAV now
        audio_path = os.path.join(audio_dir_base, audio_filename)

        try:
            # Directly save from array
            wav_arr = torch.from_numpy(audio_feature["array"]).unsqueeze(0)
            sr = audio_feature["sampling_rate"] # Already checked for presence
            torchaudio.save(audio_path, wav_arr, sr)
            # print(f"Sample {index}: Saved directly from array as WAV.") # Optional: for verbose logging
        except Exception as save_exc:
            print(f"Direct WAV save failed for sample {index}: {save_exc}")
            return None # Indicate failure to process

        # Create entry in split directory using the determined audio_filename
        split_path = os.path.join(split_dirs_base[current_split], f"{os.path.splitext(audio_filename)[0]}.txt")
        with open(split_path, "w") as f:
            f.write(f"{audio_path}\n")

        return audio_path # Indicate success by returning path

    except Exception as e:
        # This outer try-except catches errors in split determination or other unexpected issues
        print(f"Error processing sample {index} (outer try-except): {e}. Item keys: {list(item_dict.keys())}")
        return None

def download_and_extract_datasets():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download LAION dataset first
    print("\nProcessing LAION-Audio dataset...")
    laion_dir_base = os.path.join(data_dir, "laion")
    audio_dir_base = os.path.join(laion_dir_base, "audio")
    os.makedirs(audio_dir_base, exist_ok=True)

    # Create splits directories
    split_ratios = [0.98, 0.01, 0.01]
    split_names = ["train", "val", "test"]
    split_dirs_base = {name: os.path.join(laion_dir_base, name) for name in split_names}
    for d_path in split_dirs_base.values():
        os.makedirs(d_path, exist_ok=True)

1    # Stream and download all LAION samples
    print("Streaming LAION-Audio-300M samples...")
    ds = load_dataset("laion/LAION-Audio-300M", split="train", streaming=True, trust_remote_code=True)
    ds = ds.shuffle(buffer_size=10_000, seed=42)
    
    processed_count = 0
    total_samples_to_process = 220_000_000 # Process 220M samples from LAION
    # To process all, remove .take() and adjust tqdm if desired
    ds_iterable = ds.take(total_samples_to_process) 
    
    print(f"Downloading and saving LAION samples directly as WAVs...")
    with tqdm(total=total_samples_to_process, desc="Processing LAION samples") as pbar:
        for i, item in enumerate(ds_iterable):
            if save_laion_item_locally(item, i, audio_dir_base, split_dirs_base, split_ratios):
                processed_count +=1
            pbar.update(1)
            # The loop will naturally break if ds_iterable is from .take()
            # If processing full ds, you might want a counter for max_samples_to_process
            if i >= total_samples_to_process -1 and total_samples_to_process > 0 : 
                break


    print(f"\nLAION dataset processing complete! Processed {processed_count} samples as WAV.")

    # --- LibriSpeech (train-other-500) ---
    librispeech_url = 'https://www.openslr.org/resources/12/train-other-500.tar.gz'
    print("\nProcessing LibriSpeech dataset...")
    print("\nDownloading train-other-500...")
    
    tar_path = os.path.join(data_dir, "train-other-500.tar.gz")
    if not os.path.exists(tar_path):
        download_file(librispeech_url, tar_path) # download_file defined earlier
    
    print(f"\nExtracting train-other-500...")
    with tarfile.open(tar_path, bufsize=1024*1024) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting train-other-500"):
            tar.extract(member, data_dir)
    
    os.remove(tar_path)
    print("\nLibriSpeech download and extraction complete!")

if __name__ == "__main__":
    download_and_extract_datasets()
