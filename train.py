import logging
import os

import hydra
import torch.cuda
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataModule.dataModule import DataModule
from src.netModule.InpatingModule import InpatingModule

LOGGER = logging.getLogger(__name__)
@hydra.main(config_path="configs", config_name="config",version_base='1.1')
def main(config):
    config['visualizer']['save_path'] = os.path.join(os.getcwd(), config['visualizer']['save_path'])
    LOGGER.info(OmegaConf.to_yaml(config))
    OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml'))

    checkpoints_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(checkpoints_dir, exist_ok=True)
    metrics_logger = TensorBoardLogger(config['log']['tensorboard_logger_path'], name=os.path.basename(os.getcwd()))
    metrics_logger.log_hyperparams(config)

    training_model=InpatingModule(config)

    checkpoint_kwargs=config['checkpoint_kwargs']
    train_kwargs=config['train_kwargs']
    trainer = Trainer(
        # there is no need to suppress checkpointing in ddp, because it handles rank on its own
        callbacks=ModelCheckpoint(dirpath=checkpoints_dir, **checkpoint_kwargs),
        logger=metrics_logger,
        default_root_dir=os.getcwd(),
        **train_kwargs
    )
    trainer.fit(training_model,datamodule=DataModule(config))
if __name__ == '__main__':
    main()