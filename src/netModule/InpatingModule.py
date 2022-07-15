import copy
import logging

import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F

from src.netModule.Discriminator import Discriminator
from src.netModule.Evaluator import Evaluator
from src.netModule.Generator import Generator
from src.netModule.Visualizer import Visualizer

LOGGER = logging.getLogger(__name__)



class InpatingModule(ptl.LightningModule):
    def __init__(self, config,predict_only=False):
        super().__init__()
        LOGGER.info('InpaintingModule Init Called')

        self.config = config
        self.generator = Generator()

        if not predict_only:
            self.save_hyperparameters(self.config)
            self.discriminator = Discriminator()
            self.visualizer = Visualizer()
            self.val_evaluator = Evaluator()
            self.test_evaluator = Evaluator()
        LOGGER.info('InpatingModule Init Done')

    def configure_optimizers(self):
        pass


    def training_step(self, batch, batch_idx, optimizer_idx=None):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def training_step_end(self, batch_parts_outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def _do_step(self, batch, batch_idx, mode='train', optimizer_idx=None, extra_val_key=None):
        pass


    def get_current_generator(self, no_average=False):
        if not no_average and not self.training and self.average_generator and self.generator_average is not None:
            return self.generator_average
        return self.generator

    def forward(self, batch):
        pass

    def generator_loss(self, batch):
        pass



    def discriminator_loss(self, batch):
        pass

    def store_discr_outputs(self, batch):
        pass

