import os
import requests
from tqdm import tqdm

def download_model(url, save_dir):
    """Download the model checkpoint from HuggingFace"""
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Local path to save the checkpoint
    save_path = os.path.join(save_dir, "last.ckpt")
    
    # Don't download if file already exists
    if os.path.exists(save_path):
        print(f"Checkpoint already exists at {save_path}")
        return
        
    print(f"Downloading checkpoint to {save_path}...")
    
    # Send GET request with stream enabled
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(save_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
            
    print("Download complete!")

if __name__ == "__main__":
    # Model URL
    url = "https://huggingface.co/HKUSTAudio/xcodec2/resolve/main/ckpt/epoch%3D4-step%3D1400000.ckpt"
    
    # Save directory (same as used in train.py)
    save_dir = os.path.join(".", "log_dir")
    
    download_model(url, save_dir)
