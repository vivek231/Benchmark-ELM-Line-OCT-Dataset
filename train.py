import os
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from itertools import chain
from skimage.color import rgb2gray
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn as nn
import csv
from torch import optim, sigmoid 
from tqdm import tqdm
from model import LinkNetImprove, LinkNet
from torch import optim
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision
import warnings
warnings.filterwarnings("ignore")
from skimage import filters
import skimage.io as io
import argparse
from skimage.color import rgb2gray
from dataset import Dataset, RandomCrop, RandomFlip, ToTensor, Normalize
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian


parser = argparse.ArgumentParser(description='ELM_line segmentation-PyTorch-implementation')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--n_classes', type=int, default=1, help='number of classes')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
opt = parser.parse_args()

# --------------- Check GPU is available or not -------------
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

# -------------------- Model load ------------------------
n_classes = opt.n_classes
model = LinkNet(classes = n_classes)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print ("\nTrainable parameters = ", pytorch_total_params)

# -------------------- Dataloader path -----------------
OCT_train = '/home/vivek/Music/Eye_net/dataset/train/' 
OCT_val = '/home/vivek/Music/Eye_net/dataset/val/'
OCT_test = '/home/vivek/Music/Eye_net/dataset/test/'
result = '/home/vivek/Music/Eye_net/result/'

batch_size = opt.batch_size
print('===> Loading datasets')

# --------------------- Loading train and validation data from directory --------------------
transform_train = transforms.Compose([RandomFlip(), ToTensor()])
transform_val = transforms.Compose([ToTensor()])
transform_test = transforms.Compose([ToTensor()])

train_dataset = Dataset(OCT_train, data_type='float32', transform = transform_train)
val_dataset = Dataset(OCT_val, data_type='float32', transform = transform_val)
test_dataset = Dataset(OCT_test, data_type='float32',  transform = transform_test)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False) 
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False) 
num_train = len(train_loader)
num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

loss_index = []
loss_values = []
List_Loss = []

# ----------- Loss function ----
criterion_l1 = nn.L1Loss()
criterion = nn.BCEWithLogitsLoss()
params = list(model.parameters())

# -------- Hyperparameters -------
num_epochs = opt.num_epochs
learning_rate = opt.lr 
st_epoch = 0
num_epoch = 100
# ------------ Optimizer -----------
optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(opt.beta1, 0.999))
# --------- Training loop --------
for epoch in range(st_epoch + 1, num_epoch + 1):
    model.train()
    epoch_loss = []
    for i, data in enumerate(train_loader, 1):
        def should(freq):
            return freq > 0 and (i % freq == 0 or i == num_batch_train)
        input = data['input'].to(device)
        label = data['label'].to(device)
        input = Variable(input.float(), requires_grad=False)
        label = Variable(label.float(), requires_grad=False)
# ----- Forward pass--------
        out = model(input)
        out = out.squeeze(1)
        label = label.squeeze(1)
        loss = criterion(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += [loss.item()]

        print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
                % (epoch, i, num_batch_train, np.mean(epoch_loss)))

        loss_index.append(epoch)
        loss_values.append(epoch_loss) 
        List_Loss.append([epoch, epoch_loss])

        with open('Loss/Training_Loss.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(List_Loss)
        if  epoch % 1 == 0:
            torch.save(model.state_dict(),'model/Eyenet{:d}.ckpt'.format(epoch))
            dirName = result + str(epoch)
            if not os.path.exists(dirName):
                os.mkdir(dirName)
            visualization = dirName + "/"

        with torch.no_grad():
            model.eval()
            for j, data in enumerate(test_loader, 1):

                input = data['input'].to(device)
                label = data['label'].to(device)

                input = Variable(input.float(), requires_grad=False)
                label = Variable(label.float(), requires_grad=False)
                output = model(input)
                output = torch.sigmoid(output)
                output = 1.0 * (output > 0.5)
#  -------------- save all the visualiztion ---------------- 
                torchvision.utils.save_image(input, visualization+'/'+ str(j)+'_input.png')
                torchvision.utils.save_image(output, visualization+'/'+ str(j)+'_pred.png')
                torchvision.utils.save_image(label, visualization+'/'+ str(j)+'_label.png')

        plt.plot(loss_index,loss_values)
        plt.ylabel('train loss')
        plt.xlabel('Epochs')
        plt.savefig('train_loss/train_loss'+str(epoch)+'.png')
