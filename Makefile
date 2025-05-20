.PHONY: help download_rhymes download_gopipe reorganize_rhymes tsv register train

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:' Makefile | cut -d':' -f1 | sort | sed 's/^/  /'

# -----------------------------------------------------------------------------
# 1. Download processed datasets from S3
# -----------------------------------------------------------------------------
download_rhymes:
	python scripts/dnp.py \
	  --prefix processed-rhymes/ \
	  --archive processed-rhymes_dataset.tar.gz \
	  --tag rhymes \
	  --dest_root Data

# Usage: make download_gopipe

download_gopipe:
	python scripts/dnp.py \
	  --prefix processed-gopipe-movies/ \
	  --archive processed-gopipe-movies_dataset.tar.gz \
	  --tag gopipe \
	  --dest_root Data

# -----------------------------------------------------------------------------
# 2. Reorganise already-extracted rhymes folder (if needed)
# -----------------------------------------------------------------------------
reorganize_rhymes:
	python scripts/reorganize_extracted.py processed_rhymes_extracted --tag rhymes --dest_root Data

# -----------------------------------------------------------------------------
# 3. Generate TSVs for the flat Data directory
# -----------------------------------------------------------------------------
tsv:
	python scripts/get_tsv.py Data --mode flat --output_dir tsv_out

# -----------------------------------------------------------------------------
# 4. Register / package the dataset for training
# -----------------------------------------------------------------------------
register:
	python scripts/processor.py \
	  --dataset custom \
	  --input_dir Data \
	  --output_dir datasets

# -----------------------------------------------------------------------------
# 5. Kick off training (edit flags as required)
# -----------------------------------------------------------------------------
train:
python train.py \
	  --config configs/train.yaml \
	  data_dir=datasets/processed/$(DATASET) \
	  train_tsv=train.tsv \
	  val_tsv=val.tsv \
	  dataset.train.batch_size?=32 \
	  dataset.val.batch_size?=32 \
	  train.trainer.accumulate_grad_batches?=1
