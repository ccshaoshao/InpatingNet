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
    DEFAULT_KEY_ORDER = 'image predicted_image inpainted'.split(' ')

    def __init__(self, outdir='samples', key_order=DEFAULT_KEY_ORDER, max_items_in_batch=10,
                 last_without_mask=True, rescale_keys=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.key_order = key_order
        self.max_items_in_batch = max_items_in_batch
        self.last_without_mask = last_without_mask
        self.rescale_keys = rescale_keys

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        vis_img = self.visualize_mask_and_images_batch(batch, self.key_order, max_items=self.max_items_in_batch,
                                                  last_without_mask=self.last_without_mask,
                                                  rescale_keys=self.rescale_keys)

        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')

        curoutdir = os.path.join(self.outdir, f'epoch{epoch_i:04d}{suffix}')
        os.makedirs(curoutdir, exist_ok=True)
        rank_suffix = f'_r{rank}' if rank is not None else ''
        out_fname = os.path.join(curoutdir, f'batch{batch_i:07d}{rank_suffix}.jpg')

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis_img)

    def visualize_mask_and_images_batch(self,batch: Dict[str, torch.Tensor], keys: List[str], max_items=10,
                                        last_without_mask=True, rescale_keys=None) -> np.ndarray:
        batch = {k: tens.detach().cpu().numpy() for k, tens in batch.items()
                 if k in keys or k == 'mask'}

        batch_size = next(iter(batch.values())).shape[0]
        items_to_vis = min(batch_size, max_items)
        result = []
        for i in range(items_to_vis):
            cur_dct = {k: tens[i] for k, tens in batch.items()}
            result.append(self.visualize_mask_and_images(cur_dct, keys, last_without_mask=last_without_mask,
                                                    rescale_keys=rescale_keys))
        return np.concatenate(result, axis=0)

    def visualize_mask_and_images(self,images_dict: Dict[str, np.ndarray], keys: List[str],
                                  last_without_mask=True, rescale_keys=None, mask_only_first=None,
                                  black_mask=False) -> np.ndarray:
        mask = images_dict['mask'] > 0.5
        result = []
        for i, k in enumerate(keys):
            img = images_dict[k]
            img = np.transpose(img, (1, 2, 0))

            if rescale_keys is not None and k in rescale_keys:
                img = img - img.min()
                img /= img.max() + 1e-5
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)

            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif (img.shape[2] > 3):
                img_classes = img.argmax(2)
                # img = color.label2rgb(img_classes, colors=COLORS)

            if mask_only_first:
                need_mark_boundaries = i == 0
            else:
                need_mark_boundaries = i < len(keys) - 1 or not last_without_mask

            if need_mark_boundaries:
                if black_mask:
                    img = img * (1 - mask[0][..., None])
                img = mark_boundaries(img,
                                      mask[0],
                                      color=(1., 0., 0.),
                                      outline_color=(1., 1., 1.),
                                      mode='thick')
            result.append(img)
        return np.concatenate(result, axis=1)

