# X-Codec-2.0 – Architecture Guide

---

## 1. Bird's-eye view

```
(raw audio corpora) ──► data.py / processor.py ──► data/processed/<dataset>/
                                               │   ├─ audio/*.wav|mp3|flac
                                               │   ├─ train.tsv / val.tsv / test.tsv
                                               │   └─ metadata.json
                                               └─ data/datasets.json  (registry)
                                           ▲
                                           │
       PyTorch-Lightning                 DataModule  (builds loaders)
train.py ─────────► CodecLightningModule ◄────────── StandardAudioDataset / CombinedAudioDataset
```

* **data.py** (optional) bulk-downloads public datasets (LibriSpeech, LAION-Audio-300 M, …).
* **processor.py** standardises any folder of audio into a uniform layout and updates a
  central registry so the rest of the codebase needs zero special-casing.
* **data_module.py** builds PyTorch `DataLoader`s from that registry, with single- or
  multi-corpus support.
* **train.py** spins up the Lightning trainer using Hydra configs.  Checkpoints land in
  `cfg.log_dir` every 20 k steps.

---

## 2. Dataset ingestion and preprocessing

### 2.1 Downloader (`data.py`)
`python data.py` will:
1.  Download every subset of **LibriSpeech** with multithreaded `wget`.
2.  Stream **LAION-Audio-300 M** with 🤗 `datasets`, saving each MP3.  (Very large –
    edit constants or provide your own subset.)
3.  Track progress in `data/data_dl.json`.

> You **do not** have to use `data.py`.  Feel free to bring your own data folder and go
> straight to `processor.py`.

### 2.2 Processor (`processor.py`)
Run once per corpus:
```bash
python processor.py \
    --dataset librispeech \
    --input_dir ./data/LibriSpeech \
    --output_dir ./data

python processor.py \
    --dataset laion \
    --input_dir /mnt/laion_subset \
    --output_dir ./data
```
What it does:
* Converts every file to **16 kHz mono WAV** via `standardize_audio()`.
* Writes to `data/processed/<dataset>/audio/…` and records
  duration (samples) in `train/val/test.tsv`.
* Creates `metadata.json` per dataset.
* Updates **`data/datasets.json`** (the global registry).  Example entry:
```json
"librispeech": {
  "path": "processed/librispeech",
  "sample_rate": 16000,
  "audio_format": "wav",
  "stats": {"train_samples": 281241, "val_samples": 5184, "test_samples": 2620}
}
```

### 2.3 Standardised on-disk layout
```
data/
  datasets.json               # registry
  processed/
    librispeech/
      metadata.json           # above
      audio/…                 # wav/flac/mp3
      train.tsv               # <relative-path> \t <num-samples>
      val.tsv
      test.tsv
    laion/
      …
```

---

## 3. Dataset loading at training time

### 3.1 `StandardAudioDataset`
* Reads the registry, resolves the root directory, parses TSV, performs on-the-fly
  random crop / pad to `cfg.dataset.min_audio_length` (96 k samples ≈ 6 s).
* Extracts **Wav2Vec2-BERT** features (layer 0) per audio for semantic conditioning.

### 3.2 `CombinedAudioDataset`
When `config/dataset/default.yaml` lists
```yaml
dataset:
  names: [librispeech, laion]
```
the loader instantiates one `StandardAudioDataset` per name, concatenates them,
exposes a flat index space and a shared `collate_fn`.
Balancing is naïve uniform; add weighted sampling if needed.

### 3.3 `DataModule`
Wraps the dataset(s) into Lightning `DataLoader`s with per-split batch sizes:
```yaml
train:
  batch_size: 8      # *per GPU*
val:
  batch_size: 8
```

---

## 4. Model & training

### 4.1 High-level
```
Wave ► CodecEncoder ► +SemanticEncoder ► Residual FSQ ► CodecDecoder (Vocos) ► Wave
                    └────────────── fc_prior ───────┘
```
* `CodecEncoder` = dilated conv stack (hop 320).
* `CodecDecoderVocos` = Transformer + ConvNeXt backbone + ISTFT head.
* Losses: Mel (λ 15), adversarial + feature-match (HiFi-GAN MPD + SpecDisc), VQ, semantic L2.
* Optimisers: AdamW with bespoke warm-up schedule (`common/schedulers.py`).

### 4.2 Launch command
```bash
python train.py \
  dataset.names='[librispeech,laion]' \
  log_dir=./logs/mix16k_h100 \
  dataset.train.batch_size=32 dataset.val.batch_size=32
```
* `devices=4` in `config/train/default.yaml` → 4×H100.
* `ModelCheckpoint` writes `logs/mix16k_h100/epoch=0-step=20000.ckpt`, etc.
* Automatic resume from `logs/mix16k_h100/last.ckpt`.

---

## 5. Tweaking & extending

| Goal | Where to change |
|------|-----------------|
| Add new corpus | `processor.py` → `process_dataset()` handles any folder; just supply `--dataset <name>`.
| Different crop length | `cfg.dataset.min_audio_length` (samples).
| Batch size / grad accum | `cfg.dataset.*.batch_size`, `cfg.train.trainer.accumulate_grad_batches`.
| Disable/enable losses | `cfg.train.use_*_loss` flags.
| Learning-rate schedule | `cfg.train.*_schedule_params`.
| Checkpoint frequency | `every_n_train_steps` in `train.py`.

---

Happy training!  For any issue, grep the code snippets cited above or open an
issue in the repo. 