.PHONY: help download_rhymes download_gopipe format train

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
# 3. Kick off training (edit flags as required)
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

# -----------------------------------------------------------------------------
# 0. Format extracted datasets into standard layout & update registry
# -----------------------------------------------------------------------------
format:
	python scripts/format_datasets.py config/dataset/default.yaml --output_root ./data
