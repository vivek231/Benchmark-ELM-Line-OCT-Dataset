from PIL import Image, ImageOps
from torchvision.transforms import functional as F
import numpy as np
import random

"""
The code was adapted from
https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
"""

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
    
class ToTensor(object):
    def __call__(self, img, mask):
        return F.to_tensor(img), F.to_tensor(mask)

class ToPILImage(object):
    def __call__(self, img, mask):
        return F.to_pil_image(img), F.to_pil_image(mask)

# class RandomHorizontallyFlip(object):
#     def __call__(self, img, mask):
#         if random.random() < 0.5:
#             return img.transpose(Image.FLIP_LEFT_RIGHT),  mask.transpose(Image.FLIP_LEFT_RIGHT)
#         return img, mask

class Equalization(object):
    def __call__(self, img, mask):
        return ImageOps.equalize(img), mask
    
class GammaAdjustment(object):
    def __init__(self, gamma = 1.3):
        self.gamma = gamma
    def __call__(self, img, mask):
        return F.adjust_gamma(img = img, gamma = self.gamma), mask
        
class ContrastAdjustment(object):
    def __init__(self, contrast_factor = 2):
        self.contrast_factor = contrast_factor
    def __call__(self, img,mask):
        return F.adjust_contrast(img = img, contrast_factor=self.contrast_factor), mask
    
class GaussianNoise(object):
    def __call__(self, img, mask):
        return img + torch.randn_like(img), mask 
    
class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

class RandomLightVar(object):
    def __call__(self, img,mask):
        return (img+random.random()*64-32).astype('uint8'), mask

class RandomLightRevert(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return 255-img, mask
        else:
            return img, mask
        
class Normalization(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self, img,mask):
        return F.normalize(img, self.mean, self.std, self.inplace), mask
    
class Grayscale(object):
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img,mask):

        return F.to_grayscale(img, num_output_channels=self.num_output_channels), mask
        
class Resize(object):
    def __init__(self, size, interpolation = Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img,mask):
        return F.resize(img, self.size, self.interpolation), F.resize(mask, self.size, self.interpolation)    

def ELM_transform(normalize = True):
    transform = {
    'train': Compose([
        #RandomHorizontallyFlip(),
        #RandomRotate(5),
        #ContrastAdjustment(1.2),
        Resize((256,256)),
        ToTensor(),
        Normalization([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) 
    ]),
    'val': Compose([Resize((256,256)),
        ToTensor(),
        Normalization([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'test': Compose([Resize((256,256)),
        ToTensor(),
        Normalization([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    }
    return transform
