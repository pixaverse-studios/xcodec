import os
import pytorch_lightning as pl
import hydra
import torch
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from lightning_module import CodecLightningModule

@hydra.main(config_path='config', config_name='default')
def check_params(cfg):
    # Initialize model
    lightning_module = CodecLightningModule(cfg)
    
    # Set up logging
    log_dir_name = os.path.basename(os.path.normpath(cfg.log_dir))
    wandb_logger = WandbLogger(
        project='xcodec2_params',
        name=log_dir_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Check for checkpoint
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join('./log_dir/last.ckpt')
    
    if os.path.exists(ckpt_path):
        print(f"Found checkpoint at: {ckpt_path}")
        # Load checkpoint and save parameters to file
        checkpoint = torch.load(ckpt_path)
        
        with open('params.txt', 'w') as f:
            f.write("Checkpoint contents:\n")
            for key in checkpoint.keys():
                if key == 'state_dict':
                    f.write("\nModel parameters:\n")
                    for param_key in checkpoint[key].keys():
                        f.write(f"{param_key}: {checkpoint[key][param_key].shape}\n")
                else:
                    f.write(f"{key}: {type(checkpoint[key])}\n")
        
        print(f"Parameters saved to params.txt")
    else:
        print(f"No checkpoint found at {ckpt_path}")

if __name__ == '__main__':
    check_params()
