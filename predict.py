import argparse
import logging
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from efficientunet import *
from data_vis import plot_img_and_mask
from dataset_test import BasicDataset
from all_transformers import *
from model import U_Net,AttU_Net, LinkNetImprove, U2NETP,R2U_Net,DeepLabv3_plus,FCN,SegNet

#model = get_efficientunet_b3(n_classes=1, concat_input=True, pretrained=True)
#model = AttU_Net(n_channels=3, n_classes=1)
#model = U_Net(n_channels=3, n_classes=1)
#model = LinkNetImprove(n_channels=3, n_classes=1)
#model = U2NETP(n_channels=3,n_classes=1)
#model = R2U_Net(n_channels=3, n_classes=1,t=2)
#model = DeepLabv3_plus(n_channels=3, n_classes=1, os=16, pretrained=True, _print=True)
#model = FCN(n_channels=3, n_classes=1)
model = SegNet(n_channels=3, n_classes=1)

if torch.cuda.is_available():
    model.cuda()

model.load_state_dict(torch.load('/home/vivek/Documents/ELMseg/checkpoint/fold5.model'))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

image_dir = './test/image/'
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

for image_name in image_filenames:
    #model.eval()
    print (image_name)
    im = cv2.imread(image_dir + image_name)
    h,w,c = im.shape
    im = cv2.resize(im, (256,256))   
    im = im / 255.0
    im = np.expand_dims(im, 0)
    im = np.transpose(im, (0, 3, 1, 2))
    im = Variable(torch.FloatTensor(im)).cuda()

    im[:, 0, :, :] = im[:, 0, :, :] - 0.485
    im[:, 1, :, :] = im[:, 1, :, :] - 0.456
    im[:, 2, :, :] = im[:, 2, :, :] - 0.406
    # *****************************************
    im[:, 0, :, :] = im[:, 0, :, :] / 0.229
    im[:, 1, :, :] = im[:, 1, :, :] / 0.224
    im[:, 2, :, :] = im[:, 2, :, :] / 0.225
    out = model(im)
    out = torch.sigmoid(out) 
    out = out>0.5
    out = out.cpu().data.numpy()
    out = np.array(out,dtype=np.uint8)
    out = np.squeeze(out)
    out = np.expand_dims(out,0)
    out = np.transpose(out, (1, 2, 0))
    # out = cv2.resize(out,(w,h))
    print (out.shape)
    cv2.imwrite('result/'+image_name, 255*out)
