import glob
import os
import cv2
from torch.utils.data import Dataset

from src.dataModule.Mask_Generator import Mask_Generator


class TrainDataset(Dataset):
    def __init__(self, indir):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.mask_generator = Mask_Generator()
        self.index = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.mask_generator.irregular_mask(img.shape[0],img.shape[1])
        self.index += 1
        # cv2.imshow('image', img)
        # cv2.imshow('mask', mask)
        return dict(image=img,mask=mask)