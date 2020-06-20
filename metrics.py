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

def Accuracy(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 255, y_true == 255))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 255, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 255))
    accuracy = (TP + TN)/float(TP + TN + FP + FN)
    return accuracy


def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    #print(y_true.dtype)
    #print(y_pred.dtype)
    y_true = np.squeeze(y_true)/255
    y_pred = np.squeeze(y_pred)/255
    y_true.astype('bool')
    y_pred.astype('bool')
    intersection = np.logical_and(y_true, y_pred).sum()
    return ((2. * intersection.sum()) + 1.) / (y_true.sum() + y_pred.sum() + 1.)


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

def Specificity(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 255, y_true == 255))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 255, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 255))
    specificity = TN/float(TN+FP)
    return specificity

def main():
    parser = argparse.ArgumentParser(description='ELM line segmentation')
    parser.add_argument('--label_dir', type=str, default='/home/vivek/Music/demo/stack/seg',
                        help='folder of test label')
    parser.add_argument('--pred_dir', type=str, default='/home/vivek/Music/demo/stack/pred',
                        help='folder of pred masks')
    args = parser.parse_args()

    labels = [os.path.join(args.label_dir, x) for x in os.listdir(os.path.join(args.label_dir)) if 'raw' not in x]
    preds = [os.path.join(args.pred_dir, x) for x in os.listdir(os.path.join(args.pred_dir)) if 'raw' not in x]


    mean_dice = []
    mean_iou = []
    mean_sensitivity = []
    mean_specificity = []
    mean_accuracy = []    

    for l, p in zip(labels, preds):
        logging.info("Process %s and %s" % (p, l))
        G = sitk.GetArrayFromImage(sitk.ReadImage(l))
        S = sitk.GetArrayFromImage(sitk.ReadImage(p))
        mean_accuracy.append(Accuracy(G, S))
        mean_dice.append(Dice(G, S))
        mean_iou.append(IoU(G, S))
        mean_sensitivity.append(Sensitivity(G, S))
        mean_specificity.append(Specificity(G, S))

    print ('Mean_Accuracy = ', np.mean(np.array(mean_accuracy)))
    print ('Mean_Dice = ', np.mean(np.array(mean_dice)))
    print ('Mean_IoU = ', np.mean(np.array(mean_iou)))
    print ('Mean_Sensitivity = ', np.mean(np.array(mean_sensitivity)))
    print ('Mean_Specificity = ', np.mean(np.array(mean_specificity)))


if __name__ == '__main__':
    main()
