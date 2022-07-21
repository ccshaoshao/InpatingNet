import glob
import os
import logging

import numpy as np
from torch.utils.data import Dataset
import cv2
LOGGER=logging.getLogger(__name__)

class ValDataset(Dataset):
    def __init__(self, indir):
        self.datadir = indir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.replace('_mask000','') for fname in self.mask_filenames]
        LOGGER.info('ValDataset Init Done')
    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        img=cv2.imread(self.img_filenames[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype("float32") / 255
        #
        mask = cv2.imread(self.mask_filenames[i])
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        mask = np.transpose(mask, (2, 0, 1))
        mask = mask.astype("float32") / 255
        item = dict(image=img,
                      mask=mask)

        return item