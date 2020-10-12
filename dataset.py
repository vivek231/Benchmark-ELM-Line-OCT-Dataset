from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torchvision import transforms
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1,transform = None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.im_ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.mask_ids = [splitext(file)[0] for file in listdir(masks_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.im_ids)} examples')
        self.transform=transform

    def __len__(self):
        return len(self.im_ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
    

    def __getitem__(self, i):
        im_idx = self.im_ids[i]
      
        mask_idx = self.mask_ids[i]
  
        mask_file = glob(self.masks_dir + mask_idx + '.*')
        img_file = glob(self.imgs_dir + im_idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {mask_idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {im_idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {im_idx} should be the same size, but are {img.size} and {mask.size}'
        
        mask = mask.convert('1')
        img = img.convert(mode='RGB')

        if self.transform:
            img,mask=self.transform(img,mask)
      
        return {'image': img,'mask': mask}
    



    
