import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import png

def imgcolor(img,color,shape):
    img=img.reshape((-1,3))
    img=np.multiply(img, color)
    img=img.reshape((shape[0],shape[1],3))
    return img

# Read the image from the directory
dir_out='/home/vivek/Music/demo/visualize/blend/overlap/'

img_list=os.listdir('gt')
for filename in img_list:
    if filename.endswith('.png'):
        print(filename)
        img_gt=cv2.imread('gt/'+filename)
        img_gt=cv2.resize(img_gt,(384,384))
        org_img=cv2.imread('a/'+filename)
        org_img=cv2.resize(org_img,(384,384))
        img_gt=img_gt/255
        img_gt=np.array(img_gt,dtype=np.uint8)
        img_gt[np.where(img_gt<1)]=0
        img_predict=cv2.imread('/home/vivek/Music/demo/visualize/blend/out/'+filename)
        img_predict=cv2.resize(img_predict,(384,384))
        img_predict=np.array(img_predict,dtype=np.uint8)
        img_predict=img_predict/255
        img_predict=np.array(img_predict,dtype=np.uint8)
        img_predict[np.where(img_predict<1)]=0
        result=img_predict-img_gt

#   Compute the FP, TP, FN, TN *****************************************
        FP=0*img_predict
        FP[np.where(result>0)]=1
        FN=0*img_predict
        FN[np.where(result<0)]=1
        TP=0*img_predict
        TP=cv2.bitwise_and(img_predict,img_gt)
        TN=0*img_predict
        TN=cv2.bitwise_and(1-img_predict,1-img_gt)
        aa=cv2.bitwise_or(img_predict,img_gt)
#    Fill the colors into a mask ******************************************** 
        colors=[ [0, 0, 255] ,  [ 0, 255,0]  , [0, 0, 0],  [0, 0, 0]]   # Red, Yellow, Green,Blue
        colors=np.array(colors,dtype=np.uint8 )
        shape=img_gt.shape
        img_gt=imgcolor(img_gt,colors[0],shape)
        img_predict=imgcolor(img_predict,colors[1],shape)
        FP=imgcolor(FP,colors[2],shape)
        TP=imgcolor(TP,colors[3],shape)
#   Image Blending opearation ********************************************
        dst1 = cv2.addWeighted(FP,0.5,TP,0.5,0)
        Blend_org = cv2.addWeighted(img_gt,0.8,  img_predict,0.5,0)
        Blend_org = cv2.addWeighted(Blend_org,0.8, org_img,1.0,0)
        cv2.imwrite(dir_out+filename, Blend_org)
