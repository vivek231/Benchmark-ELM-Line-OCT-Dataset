import os
import numpy as np
import skimage.io
from skimage import measure
from skimage.transform import rotate
import matplotlib.pyplot as plt

def display(im):
  # create figure
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)
  # add image
  cmap = plt.cm.get_cmap('jet')
  # cmap.set_bad(color='black') # does not work - fix it
  # ax.imshow(im, cmap=cmap, interpolation='none')
  cmap.set_under(color='black')
  p = ax.imshow(im, cmap=cmap, interpolation='none', vmin=0.0000001)
  plt.colorbar(p)
  plt.show()

def centroid_2d(x, y):
  xc = np.mean(x)
  yc = np.mean(y)
  return xc, yc

def find_left_2d(im, d=10):
  im = im[:,0:d]
  idx = np.asarray(np.where(im == 1))
  x = idx[1,:]
  y = idx[0,:]
  return x, y

def find_right_2d(im, d=10):
  s = im.shape[1]
  im = im[:,s-d:s]
  idx = np.asarray(np.where(im == 1))
  x = s - d + idx[1,:]
  y = idx[0,:]
  return x, y

def angle_between_vectors(vector_1, vector_2):
  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  angle = np.arccos(dot_product)
  angle = np.rad2deg(angle)
  return angle

if __name__ == '__main__':
  
  # parameters
  im_root = '../IMAGES/'
  im_names = [f for f in os.listdir(im_root) if f.endswith('_seg.tif')]

  # max image shape in the datasets
  ms = (0, 0, 0)
  for im_name in im_names:
    # read image
    im_path = os.path.join(im_root, im_name)
    im = skimage.io.imread(im_path)
    im = np.moveaxis(im, 0, -1)  # [z,x,y] -> [x,y,z]
    ms = max(ms, im.shape)

  # shapes summary
  imp = np.zeros(ms, dtype=float)
  for im_name in im_names:
    # read image
    im_path = os.path.join(im_root, im_name)
    im = skimage.io.imread(im_path)
    im = np.moveaxis(im, 0, -1)  # [z,x,y] -> [x,y,z]
    im = im==255 # binary

    # align lines
    yc = int(im.shape[0]/2)
    yd = int(im.shape[0]/3)
    for z in range(0,im.shape[2]-1):
      # line ends
      imz = im[:,:,z]
      xl, yl = find_left_2d(imz)
      xr, yr = find_right_2d(imz)
      xlc, ylc = centroid_2d(xl, yl)
      xrc, yrc = centroid_2d(xr, yr)
      # print([xlc, ylc, xrc, yrc])

      # line angle
      v1 = [xrc-xlc, yrc-ylc]
      v2 = [1-0, 0-0]
      a = angle_between_vectors(v1, v2)
      # print(a)

      # rotate
      imzr = rotate(imz, a)
      imzr = imzr.astype(np.bool)
      xl, yl = find_left_2d(imzr)
      xr, yr = find_right_2d(imzr)
      xlc, ylc = centroid_2d(xl, yl)
      xrc, yrc = centroid_2d(xr, yr)
      xlc = int((xlc+xrc)/2)
      ylc = int((ylc+yrc)/2)

      # add all shapes up
      imzr = imzr.astype(np.float)
      imp[yc-yd:yc+yd,0:im.shape[1],z] = imp[yc-yd:yc+yd,0:im.shape[1],z] + imzr[ylc-yd:ylc+yd,:]

  # add all shapes up
  imp = np.sum(imp, axis=2)

  # display image
  display(imp)