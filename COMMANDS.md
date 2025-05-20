# Dataset & Training Commands Cheat-Sheet

Use this as a quick reference for every step – from fetching raw data to kicking off model training.

---

## 1. Download processed archives from S3 and unpack

```bash
# Rhymes dataset
python scripts/dnp.py \
  --prefix processed-rhymes/ \
  --archive processed-rhymes_dataset.tar.gz \
  --tag rhymes \
  --dest_root Data

# GoPipe-Movies dataset
python scripts/dnp.py \
  --prefix processed-gopipe-movies/ \
  --archive processed-gopipe-movies_dataset.tar.gz \
  --tag gopipe \
  --dest_root Data
```

The commands above will:
1. Skip the download if the archive already exists locally (size-checked).
2. Show a progress bar for both download & extraction.
3. Copy all audio into
   ```
   Data/train/audios
   Data/val/audios
   Data/test/audios (if present)
   ```
   with unique filenames prefixed by the provided `--tag`.

---

## 2. Re-organise an already extracted directory
(Useful when the JSON splits are missing.)

```bash
python scripts/reorganize_extracted.py processed_rhymes_extracted --tag rhymes --dest_root Data
```

Scans for every `*.wav|*.flac|*.mp3`, performs a 90/10 train/val split, then copies the files into the same flat structure as above.

---

## 3. Generate TSV files for fairseq / wav2vec-style training

```bash
python scripts/get_tsv.py Data --mode flat --output_dir tsv_out
```

Creates:
```
./tsv_out/
  Data_train.tsv
  Data_val.tsv
  Data_test.tsv   # only if test/audios exists
```
Each line contains `<relative_path>\t<num_samples>` with NaN checks.

---

## 4. Register / package the dataset

```bash
python scripts/processor.py \
  --dataset custom \
  --input_dir Data \
  --output_dir datasets
```

Produces `datasets/processed/<dataset_name>/` with
```
metadata.json
train.tsv  val.tsv  test.tsv
audio/
```
and updates (or creates) `datasets/datasets.json` registry.

---

## 5. Start training (PyTorch-Lightning example)

> Replace CONFIG with your actual configuration file.

```bash
python train.py \
  --config configs/train.yaml \
  data_dir=datasets/processed/<dataset_name> \
  train_tsv=train.tsv \
  val_tsv=val.tsv
```

If you are using hydra-style configs:
```bash
python train.py +dataset.name=<dataset_name>
```

---

## 6. Quick sanity check

```bash
python - << 'PY'
import torchaudio, random, glob
files = glob.glob('Data/train/audios/*.wav')
fp = random.choice(files)
w, sr = torchaudio.load(fp)
print(fp, w.shape, sr)
PY
```

---

Feel free to adapt paths / flags as required by your environment or training pipeline. 