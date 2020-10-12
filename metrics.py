import os
import argparse
import logging
import numpy as np
import SimpleITK as sitk
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm
import cv2
import sys
from PIL import Image
from sklearn import metrics
from scores import *
import numpy as np
#from sewar.full_ref import rmse, mse


np.seterr(divide='ignore', invalid='ignore')

def Accuracy(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 255, y_true == 255))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 255, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 255))
    accuracy = (TP + TN)/float(TP + TN + FP + FN)
    return accuracy


def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    TP = np.sum(np.logical_and(y_pred == 255, y_true == 255))
    TP = 2*TP
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 255, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 255))
    dice = TP/float(TP + FP + FN)

    return dice



# def Dice(y_true, y_pred):
#     """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
#     #print(y_true.dtype)
#     #print(y_pred.dtype)
#     y_true = np.squeeze(y_true)/255
#     y_pred = np.squeeze(y_pred)/255
#     y_true.astype('bool')
#     y_pred.astype('bool')
#     intersection = np.logical_and(y_true, y_pred).sum()
#     return ((2. * intersection.sum()) + 1.) / (y_true.sum() + y_pred.sum() + 1.)


def IoU(Gi,Si):
    #print(Gi.shape, Si.shape)
    Gi = np.squeeze(Gi)/255
    Si = np.squeeze(Si)/255
    Gi.astype('bool')
    Si.astype('bool')
    intersect = 1.0*np.sum(np.logical_and(Gi,Si))
    union = 1.0*np.sum(np.logical_or(Gi,Si))
    return intersect/union

def Sensitivity(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 255, y_true == 255))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 255, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 255))
    sensitivity = TP/float(TP + FN)
    return sensitivity

def Precision(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 255, y_true == 255))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 255, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 255))
    precision = TP/float(TP + FP)
    return precision

def Specificity(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 255, y_true == 255))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 255, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 255))
    specificity = TN/float(TN+FP)
    return specificity

from PIL import Image # No need for ImageChops
import math
from skimage import img_as_float
from skimage.measure import compare_mse as mse
from sklearn.metrics import mean_absolute_error,mean_squared_error

def rmsdiff(im1, im2):
    im1 = im1/255
    im2 = im2/255
    """Calculates the root mean square error (RSME) between two images"""
    return math.sqrt(mse(img_as_float(im1), img_as_float(im2)))

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def main():
    parser = argparse.ArgumentParser(description='ELM line segmentation accuracy computation')
    parser.add_argument('--label_dir', type=str, default='/home/vivek/Documents/ELMseg/accuracy/fold5/gt/',
                        help='folder of test label')
    parser.add_argument('--pred_dir', type=str, default='/home/vivek/Documents/ELMseg/Final_Results/256*256_without_data_augment/segnet/fold5/',
                        help='folder of pred masks')
    args = parser.parse_args()

    labels = [os.path.join(args.label_dir, x) for x in os.listdir(os.path.join(args.label_dir)) if 'raw' not in x]
    preds = [os.path.join(args.pred_dir, x) for x in os.listdir(os.path.join(args.pred_dir)) if 'raw' not in x]

    mean_dice = []
    mean_iou = []
    mean_sensitivity = []
    mean_specificity = []
    mean_false_positive_rate = []
    mean_accuracy = []    
    mean_hausdroff = []
    mean_mse = []
    mean_mae = []
    mean_rmse = []

    for l, p in zip(labels, preds):
        # logging.info("Process %s and %s" % (p, l))
        G = sitk.GetArrayFromImage(sitk.ReadImage(l))
        S = sitk.GetArrayFromImage(sitk.ReadImage(p))
        mean_accuracy.append(Accuracy(G, S))
        mean_dice.append(Dice(G, S))
        mean_iou.append(IoU(G, S))
        mean_sensitivity.append(Sensitivity(G, S))
        mean_specificity.append(Specificity(G, S))
        mean_false_positive_rate.append(false_positive_rate(S, G))
        mean_hausdroff.append(hausdorff_distance(G, S))
        mean_mse.append(mean_squared_error(G,S))
        mean_mae.append(mean_absolute_error(G,S))
        mean_rmse.append(rmsdiff(G,S))

    print ('!---------------------------------------------!')
    print ('!   Metrices         !   Mean scores')
    print ('!---------------------------------------------!')
    print ('! Accuracy           !',  100*np.mean(np.array(mean_accuracy)))
    print ('!---------------------------------------------!')
    print ('! Dice coefficient   !',  100*np.mean(np.array(mean_dice)))
    print ('!---------------------------------------------!')
    print ('! IoU                !',  100*np.mean(np.array(mean_iou)))
    print ('!---------------------------------------------!')
    print ('! Sensitivity        !',  100*np.mean(np.array(mean_sensitivity)))
    print ('!---------------------------------------------!')
    print ('! Specificity        !',  100*np.mean(np.array(mean_specificity)))
    print ('!---------------------------------------------!')
    print ('! False positive rate!',  100*np.mean(np.array(mean_false_positive_rate)))
    print ('!---------------------------------------------!')
    print ('! Mean squared error !',  np.mean(np.array(mean_mse)))
    print ('!---------------------------------------------!')
    print ('! Mean absolute error !',  np.std(np.array(mean_mae)))
    print ('!---------------------------------------------!')
    print ('! Root mean absolute error !', np.mean(np.array(mean_rmse)))
    print ('!---------------------------------------------!')
    print ('! Hausdroff distance !',  np.mean(np.array(mean_hausdroff)))
    print ('!---------------------------------------------!')


if __name__ == '__main__':
    main()
