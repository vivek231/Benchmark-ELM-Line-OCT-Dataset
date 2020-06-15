import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir_data = '/home/vivek/Downloads/pytorch-UNET-master_tif/pytorch-UNET-master/datasets/em'
dir_save_train = os.path.join(dir_data, 'train')
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

for filename in os.listdir(dir_data):
    if filename.endswith("_seg.tif"):
        dataset = Image.open(os.path.join(dir_data, filename))
        h,w =  np.shape(dataset)
        tiffarray = np.zeros((h,w,dataset.n_frames))
        for j in range(dataset.n_frames):
            dataset.seek(j)
            target_ = np.asarray(dataset)
            filename = filename.split('_seg.tif')[0]
            np.save(os.path.join(dir_save_train,  filename + "_%02d.npy" % j), (target_))


input = np.load('/home/vivek/Downloads/pytorch-UNET-master_tif/pytorch-UNET-master/datasets/em/train/10. goodwin images need_20.npy')
print (input.shape)

plt.subplot(121)
plt.imshow(input)

plt.show()
