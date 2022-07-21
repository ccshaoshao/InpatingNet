import glob
import logging
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

from src.dataModule.Mask_Generator import Mask_Generator
LOGGER = logging.getLogger(__name__)

class TrainDataset(Dataset):
    def __init__(self, indir):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.mask_generator = Mask_Generator()
        self.index = 0
        LOGGER.info('TrainDataset Init Done')
    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype("float32") / 255
        #
        mask = self.mask_generator.irregular_mask(img.shape[1],img.shape[2])
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        mask = np.transpose(mask, (2, 0, 1))
        #mask = mask.astype("float32") / 255
        self.index += 1
        # cv2.imshow('image', img)
        # cv2.imshow('mask', mask)
        return dict(image=img,mask=mask)