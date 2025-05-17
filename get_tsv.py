from multiprocessing import Pool, cpu_count
import os
import torchaudio
import torch
from tqdm import tqdm

def process_file(args):
    file_path, root_dir = args  
    try:
        rel_path = os.path.relpath(file_path, start=root_dir)
        waveform, sample_rate = torchaudio.load(file_path)
        nsample = waveform.shape[1]
        batch_size = 10000
        for start in range(0, waveform.numel(), batch_size):
            end = min(start + batch_size, waveform.numel())
            if torch.isnan(waveform.view(-1)[start:end]).any():
                print(f"Skipping {rel_path} - contains NaN values")
                return None

        if nsample == 0:
            print(f"Skipping {rel_path} - zero length")
            return None  
        return f"{rel_path}\t{nsample}\n"
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def list_audio_files(root_dir, output_file, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []
    
    audio_files = []
    for root, dirs, files in os.walk(root_dir):
        # 排除指定的子文件夹
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in exclude_dirs]
        
        for filename in files:
            if filename.endswith(('.wav', '.flac', '.mp3')):
                file_path = os.path.join(root, filename)
                audio_files.append((file_path, root_dir))

    # 按文件名排序
    audio_files.sort(key=lambda x: x[0])

    # 使用多进程处理文件
    pool = Pool(processes=max(1, int(cpu_count() / 2)))
    results = list(tqdm(pool.imap(process_file, audio_files), total=len(audio_files), desc="Processing audio files"))
    pool.close()
    pool.join()

    # 写入结果到文件
    with open(output_file, 'w') as file:
        # Write only the paths, no header line with root directory
        for result in results:
            if result:
                file.write(result)

if __name__ == "__main__":
    librispeech_base_dir = "./data/LibriSpeech"  # Base directory where data.py extracts LibriSpeech
    
    splits = {
        "train": "train-clean-100",
        "dev": "dev-clean",
        "test": "test-clean"
    }

    os.makedirs("./data", exist_ok=True)

    for split_name, split_folder in splits.items():
        root_directory = os.path.join(librispeech_base_dir, split_folder)
        output_tsv = os.path.join("./data", f"librispeech_{split_name}.tsv")
        
        if os.path.exists(root_directory):
            print(f"Processing LibriSpeech {split_name} split from: {root_directory}")
            list_audio_files(root_directory, output_tsv, exclude_dirs=None)
            print(f"Finished processing. TSV file saved to: {output_tsv}")
        else:
            print(f"Directory not found for LibriSpeech {split_name} split: {root_directory}")
            print(f"Make sure you have downloaded LibriSpeech datasets to {librispeech_base_dir}")
    
    print("\nFinished generating TSV files for LibriSpeech.")