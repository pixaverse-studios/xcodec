import os
import wget
import tarfile
from tqdm import tqdm

def download_and_extract_librispeech():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    urls = {
        'train-clean-100': 'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
        'dev-clean': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
        'test-clean': 'https://www.openslr.org/resources/12/test-clean.tar.gz'
    }

    # Download and extract each subset
    for subset, url in urls.items():
        print(f"\nDownloading {subset}...")
        
        # Download file
        tar_path = os.path.join(data_dir, f"{subset}.tar.gz")
        if not os.path.exists(tar_path):
            wget.download(url, tar_path)
        
        # Extract file
        print(f"\nExtracting {subset}...")
        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc=f"Extracting {subset}"):
                tar.extract(member, data_dir)
        
        # Remove tar file after extraction
        os.remove(tar_path)

    print("\nDownload and extraction complete!")

if __name__ == "__main__":
    download_and_extract_librispeech()
