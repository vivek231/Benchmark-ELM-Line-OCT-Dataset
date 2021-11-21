'''
This code is written by:

Dr. Vivek Kumar Singh
Department of Computer Science
Newcastle University, United Kingdom
Date: 24/August/2021

Also, thanks to "https://github.com/milesial/" for utilzing some of their codes.

'''
import argparse
import logging
import os
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from eval import eval_net
from model import U_Net,AttU_Net,LinkNetImprove,U2NETP,R2U_Net,DeepLabv3_plus,FCN,SegNet
from transformation import ELM_transform
from tensorboardX import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from dice_loss import Dice_Loss
import torch.nn.functional as F
from efficientunet import *
import matplotlib.pyplot as plt
import csv

#------------- Load the images from directory----------
train_dir_img = './train/image/'
train_dir_mask = './train/mask/'
val_dir_img = './val/image/'
val_dir_mask = './val/mask/'
dir_checkpoint = './checkpoint/'
# ------------------Loading END ------------------------


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

n_classes =1
n_channels = 3

def train_net(net,
              device,
              epochs=15,
              batch_size=4,
              lr=0.0001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1):
    
    transform = ELM_transform()
    train_dataset = BasicDataset(train_dir_img, train_dir_mask, img_scale,transform = transform['train'])
    
    val_dataset= BasicDataset(val_dir_img, val_dir_mask, img_scale,transform['val'])
    n_train=len(train_dataset)
    n_val=len(val_dataset)

# ----------- Load the dataset from the directory---------

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

# !!---------- Defined the optimizer --------------------------!!

    optimizer = optim.Adam(net.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=10)

# ------------------ Loss function ------------------!!
    criterion_dice = Dice_Loss
    criterion = nn.BCEWithLogitsLoss().cuda()
    
# !!-------------- Training and validation loop ------------------!!
    loss_index = []
    loss_values = []
    List_Loss = []
    best_acc=0
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                out_new =  F.sigmoid(masks_pred)

                # masks_probs_flat = masks_pred.view(-1)
                # true_masks_flat = true_masks.view(-1)
                loss = 0.5*criterion(masks_pred, true_masks) + 0.5*criterion_dice(out_new, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                loss_index.append(epoch) 
                loss_values.append(epoch_loss)   
                List_Loss.append([epoch, epoch_loss])

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % ((n_val+n_train) // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                    
                    if val_score > best_acc:
                        best_acc = val_score
                        best_model_wts = copy.deepcopy(net.state_dict())

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

    if save_cp:
        try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        net.load_state_dict(best_model_wts)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        torch.save(net.state_dict(), '{}ELM_{}.model'.format(dir_checkpoint,timestamp))

    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='2D ELM line segmentation from OCT images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=15.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')   

# -------------- Load the model -----------
    #net = get_efficientunet_b3(n_classes=1, concat_input=True, pretrained=True)
    #net = LinkNetImprove(n_channels=3, n_classes=1)
    #net = AttU_Net(n_channels=3, n_classes=1)
    #net = U_Net(n_channels=3, n_classes=1)
    #net = R2U_Net(n_channels=3, n_classes=1,t=2)
    #net = DeepLabv3_plus(n_channels=3, n_classes=1, os=16, pretrained=True, _print=True)
    #net = FCN(n_channels=3, n_classes=1)
    net = SegNet(n_channels=3, n_classes=1)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
