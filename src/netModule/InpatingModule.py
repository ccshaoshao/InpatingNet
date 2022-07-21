import logging
from typing import Optional
import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from src.netModule.discriminatorModule.Discriminator import NLayerDiscriminator
from src.netModule.evaluatorModule.Evaluator import Evaluator,InpaintingEvaluator
from src.netModule.generatorModule.ffc import FFCResNetGenerator
from src.netModule.evaluatorModule.Visualizer import Visualizer,DirectoryVisualizer
from src.netModule.lossModule.adversarial import BCELoss, NonSaturatingWithR1

LOGGER = logging.getLogger(__name__)



class InpatingModule(ptl.LightningModule):
    def __init__(self, config,predict_only=False):
        super().__init__()
        LOGGER.info('InpaintingModule Init Called')

        self.config = config
        self.generator = FFCResNetGenerator(**self.config['generator'])

        if not predict_only:
            self.save_hyperparameters(self.config)
            self.discriminator = NLayerDiscriminator()
            self.visualizer = DirectoryVisualizer(self.config['visualizer']['save_path'])
            self.val_evaluator = InpaintingEvaluator()
            self.test_evaluator = InpaintingEvaluator()
            self.adversarial_loss=NonSaturatingWithR1()
        LOGGER.info('InpatingModule Init Done')

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']

        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)

        batch['predicted_image'] = self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

        return batch

    def training_step(self, batch, batch_idx,optimizer_idx):
        if optimizer_idx == 0:  # step for generator
            for param in self.generator.parameters():
                param.requires_grad=True
            for param in self.discriminator.parameters():
                param.requires_grad=False
        elif optimizer_idx == 1:  # step for discriminator
            for param in self.generator.parameters():
                param.requires_grad = True
            for param in self.discriminator.parameters():
                param.requires_grad = False
        batch = self(batch)
        total_loss = 0
        metrics = {}
        if optimizer_idx is None or optimizer_idx == 0:  # step for generator
            total_loss, metrics = self.generator_loss(batch)

        elif optimizer_idx is None or optimizer_idx == 1:  # step for discriminator
            total_loss, metrics = self.discriminator_loss(batch)

        result = dict(loss=total_loss, log_info={'train_'+k:v for k,v in metrics.items()})
        if batch_idx%100==0:
            self.visualizer.visualize(self.current_epoch, batch_idx, batch)

        return result

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        full_loss = (step_output['loss'].mean()
                     if torch.is_tensor(step_output['loss'])  # loss is not tensor when no discriminator used
                     else torch.tensor(step_output['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in step_output['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)
        return full_loss

    def validation_step(self, batch, batch_idx):
        pass
    def validation_epoch_end(self, outputs:dict):
        pass
    def configure_optimizers(self):
        return [
            dict(optimizer=torch.optim.Adam(self.generator.parameters(), **self.config.optimizers.generator)),
            dict(optimizer=torch.optim.Adam(self.discriminator.parameters(), **self.config.optimizers.discriminator)),
        ]

    def get_current_generator(self):
        return self.generator

    def generator_loss(self, batch):
        img = batch['image']
        img=img.float()
        predicted_img = batch['predicted_image']
        original_mask = batch['mask']

        l1_value = F.l1_loss(predicted_img, img, reduction='none')

        total_loss = l1_value

        metrics = dict(gen_l1=l1_value)


        # discriminator
        # adversarial_loss calls backward by itself
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred,
                                                                         mask=original_mask)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss

        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}
        img=batch['image']
        img=img.float()
        mask=batch['image']
        predicted_img = batch['predicted_image'].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=img, fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=img,
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=mask)
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update({'adv_'+k:v for k,v in metrics.items()})

        return total_loss, metrics
