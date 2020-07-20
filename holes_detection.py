import os
import numpy as np
import skimage.io
from skimage import measure
import matplotlib.pyplot as plt

def display(im):
  plt.imshow(im, cmap='jet', interpolation='nearest')
  plt.show()

if __name__ == '__main__':
  
  # parameters
  im_root = '../IMAGES/'
  im_name = 'EBU_seg2.tif'

  # read image
  im_path = os.path.join(im_root, im_name)
  im = skimage.io.imread(im_path)
  im = np.moveaxis(im, 0, -1)  # [z,x,y] -> [x,y,z]

  # max
  im = np.amax(im, axis=0)

  # binary
  im = im==0

  # nr of abels
  imlabels = measure.label(im)
  print('Nr. of labels: ' + str(imlabels.max()))

  # display image
  display(imlabels)