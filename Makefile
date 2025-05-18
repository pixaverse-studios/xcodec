python train.py \
    dataset.train.batch_size=32 \
    dataset.val.batch_size=32 \
    train.trainer.accumulate_grad_batches=1   # 128 global batch
