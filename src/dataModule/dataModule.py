from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader


from src.dataModule.TrainDataset import TrainDataset
from src.dataModule.ValDataset import ValDataset


class DataModule(LightningDataModule):
    def __init__(self,config):
        super(DataModule, self).__init__()
        self.train_dataset_config=config['train_dataset']
        self.val_dataset_config=config['val_dataset']

    def train_dataloader(self):
        path=self.train_dataset_config['path']
        batch_size=self.train_dataset_config['batch_size']
        return DataLoader(TrainDataset(path),batch_size=batch_size)

    def val_dataloader(self):
        paths=self.val_dataset_config['paths']
        batch_size=self.val_dataset_config['batch_size']
        loaders={}
        for key in paths:
            path=paths[key]
            loaders[key]= DataLoader(ValDataset(path),batch_size=batch_size)
        combined_loaders = CombinedLoader(loaders)
        return combined_loaders
