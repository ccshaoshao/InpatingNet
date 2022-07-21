from typing import Dict, List

import cv2
import numpy as np
import os

import torch
from skimage.segmentation import mark_boundaries
from skimage import color


class Visualizer():
    DEFAULT_KEY_ORDER = 'image predicted_image inpainted'.split(' ')

    def __init__(self):
        pass

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        pass

class DirectoryVisualizer():

    def __init__(self, outdir='samples', max_items_in_batch=10,
                 last_without_mask=True, rescale_keys=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.max_items_in_batch = max_items_in_batch
        self.last_without_mask = last_without_mask
        self.rescale_keys = rescale_keys

    def visualize(self,epoch_i, batch_i, batch:dict):
        batch = {k: tens.detach().cpu().numpy() for k, tens in batch.items()}
        masks = batch['mask'] > 0.5

        batch_size=len(masks)
        groups_image=[]
        for group in range(batch_size):
            group_images=[]
            for key in batch:
                img=batch[key][group]
                if len(img.shape) == 2:
                    img = np.expand_dims(img, 2)
                img = np.transpose(img, (1, 2, 0))
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                img = mark_boundaries(img,
                                      masks[group][0],
                                      color=(1, 0, 0),
                                      outline_color=(1, 1, 1),
                                      mode='thick')
                group_images.append(img)
            group_image=np.concatenate(group_images, axis=1)
            groups_image.append(group_image)
        vis_img=np.concatenate(groups_image,axis=0)
        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')

        curoutdir = os.path.join(self.outdir, f'epoch{epoch_i:04d}')
        os.makedirs(curoutdir, exist_ok=True)
        out_fname = os.path.join(curoutdir, f'batch{batch_i:07d}.jpg')

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis_img)

