import os
 
import pytorch_lightning as pl
import hydra
import torch
import random
import time
from os.path import join, basename, exists
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy,FSDPStrategy
from torch.utils.data import DataLoader
from data_module import DataModule as LocalDataModule
try:
    from new_module import DataModule as StreamingDataModule  # noqa: E402
except ImportError:
    StreamingDataModule = None
from lightning_module import CodecLightningModule
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
 
seed = 1024
seed_everything(seed)
 
@hydra.main(config_path='config', config_name='default')
def train(cfg):
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.log_dir, 
                            save_top_k=-1, save_last=True,
                            every_n_train_steps=20000, monitor='mel_loss', mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    # Choose DataModule implementation (local files vs streaming)
    # We import both and decide inside `train()` based on cfg.dataset.streaming
    streaming_flag = cfg.dataset.get("streaming", True)
    datamodule = LocalDataModule(cfg) if (not streaming_flag or StreamingDataModule is None) else StreamingDataModule(cfg)
    lightning_module = CodecLightningModule(cfg)
    log_dir_name = os.path.basename(os.path.normpath(cfg.log_dir))
    wandb_logger = WandbLogger(
        project='xcodec2',  # 替换为您的项目名称
        name=log_dir_name,              # 替换为您的运行名称
        config=OmegaConf.to_container(cfg, resolve=True)  # 将 Hydra 配置转换为字典并传递
    )    

    # Determine checkpoint path for resumption
    # ModelCheckpoint saves to cfg.log_dir and save_last=True creates last.ckpt
    # So, the resume path should be derived from cfg.log_dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change to root directory where train.py is located

    potential_last_ckpt = os.path.join('./log_dir/last.ckpt')

    print("="*80)
    print("CURRENT DIRECTORY:", os.getcwd())
    print("LOOKING FOR CHECKPOINT AT:", potential_last_ckpt)
    print("="*80)
    actual_ckpt_to_load = None # This will be passed to trainer.fit
    if os.path.exists(potential_last_ckpt):
        actual_ckpt_to_load = potential_last_ckpt
        print("\n" + "="*80)
        print("###  RESUMING TRAINING FROM THE LATEST CHECKPOINT  ###")
        print(f"###  {actual_ckpt_to_load}  ###")
        print("="*80 + "\n")
    else:
        print(f"No checkpoint found at {potential_last_ckpt}, starting training from scratch.")

    trainer = pl.Trainer(
        **cfg.train.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=wandb_logger,
        profiler="simple",  # 启用 Profiler
        limit_train_batches=1.0 if not cfg.debug else 0.001
    )
    torch.backends.cudnn.benchmark = True  
    # lightning_module.strict_loading = False
    # LightningModule.strict_loading = True
    trainer.fit(lightning_module, datamodule=datamodule,ckpt_path=actual_ckpt_to_load)
    print(f'Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}')

if __name__ == '__main__':
    train()
