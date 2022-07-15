import glob
import os

from torch.utils.data import Dataset
import cv2

class ValDataset(Dataset):
    def __init__(self, indir):
        self.datadir = indir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.replace('_mask000','') for fname in self.mask_filenames]

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        item = dict(image=cv2.imread(self.img_filenames[i]),
                      mask=cv2.imread(self.mask_filenames[i]))
        # cv2.imshow('image',item['image'])
        # cv2.imshow('mask',item['mask'])
        # cv2.waitKey(0)
        return item