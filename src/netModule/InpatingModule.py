import logging

import pytorch_lightning as ptl
import torch
import torch.nn.functional as F

from src.netModule.discriminatorModule.Discriminator import NLayerDiscriminator
from src.netModule.evaluatorModule.Evaluator import Evaluator,InpaintingEvaluator
from src.netModule.generatorModule.ffc import FFCResNetGenerator
from src.netModule.evaluatorModule.Visualizer import Visualizer,DirectoryVisualizer

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
            self.visualizer = DirectoryVisualizer()
            self.val_evaluator = InpaintingEvaluator()
            self.test_evaluator = InpaintingEvaluator()
        LOGGER.info('InpatingModule Init Done')

    def configure_optimizers(self):
        return [
            dict(optimizer=torch.optim.Adam(self.generator.parameters(), **self.config.optimizers.generator)),
            dict(optimizer=torch.optim.Adam(self.discriminator.parameters(), **self.config.optimizers.discriminator)),
        ]


    def training_step(self, batch, batch_idx,optimizer_idx):
        if optimizer_idx == 0:  # step for generator
            set_requires_grad(self.generator, True)
            set_requires_grad(self.discriminator, False)
        elif optimizer_idx == 1:  # step for discriminator
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, True)

        batch = self(batch)

        total_loss = 0
        metrics = {}

        if optimizer_idx is None or optimizer_idx == 0:  # step for generator
            total_loss, metrics = self.generator_loss(batch)

        elif optimizer_idx is None or optimizer_idx == 1:  # step for discriminator
            if self.config.losses.adversarial.weight > 0:
                total_loss, metrics = self.discriminator_loss(batch)

        if self.get_ddp_rank() in (None, 0) and (batch_idx % self.visualize_each_iters == 0 or mode == 'test'):
            if self.config.losses.adversarial.weight > 0:
                if self.store_discr_outputs_for_vis:
                    with torch.no_grad():
                        self.store_discr_outputs(batch)
            vis_suffix = f'_{mode}'
            if mode == 'extra_val':
                vis_suffix += f'_{extra_val_key}'
            self.visualizer(self.current_epoch, batch_idx, batch, suffix=vis_suffix)

        metrics_prefix = f'{mode}_'
        if mode == 'extra_val':
            metrics_prefix += f'{extra_val_key}_'
        result = dict(loss=total_loss, log_info=add_prefix_to_keys(metrics, metrics_prefix))
        if mode == 'val':
            result['val_evaluator_state'] = self.val_evaluator.process_batch(batch)
        elif mode == 'test':
            result['test_evaluator_state'] = self.test_evaluator.process_batch(batch)
        elif mode == 'extra_val':
            result[f'extra_val_{extra_val_key}_evaluator_state'] = self.extra_evaluators[extra_val_key].process_batch(
                batch)

        return result

    def validation_step(self, batch, batch_idx):
        pass

    def training_step_end(self, batch_parts_outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def get_current_generator(self):
        return self.generator

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']

        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)

        batch['predicted_image'] = self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        return batch

    def generator_loss(self, batch):
        pass



    def discriminator_loss(self, batch):
        pass

    def store_discr_outputs(self, batch):
        pass

