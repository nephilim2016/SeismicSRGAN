#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:59:36 2021

@author: nephilim
"""

import numpy as np
import keras
import skimage.transform
import my_Im2col
import DataNormalized
from matplotlib import pyplot,cm



def Patch2TrainData(Patch,PatchSize):
    TrainData=np.zeros(((Patch.shape[1],)+PatchSize+(1,)))
    for idx in range(Patch.shape[1]):
        TrainData[idx,:,:,0]=Patch[:,idx].reshape(PatchSize)
    return TrainData

def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def PreProcess_Data(Data):
    PreData=np.zeros_like(Data)
    MaxData=[]
    MinData=[]
    for idx in range(Data.shape[0]):
        Data_tmp,MaxData_tmp,MinData_tmp=DataNormalized.DataNormalized(Data[idx,:,:,0])
        PreData[idx,:,:,0]=Data_tmp
        MaxData.append(MaxData_tmp)
        MinData.append(MinData_tmp)
    return MaxData,MinData,PreData

def PostProcess_Data(Data,MaxData,MinData):
    PostData=np.zeros_like(Data)
    for idx in range(Data.shape[0]):
        Data_tmp=DataNormalized.InverseDataNormalized(Data[idx,:,:],MaxData[idx],MinData[idx])
        PostData[idx,:,:]=Data_tmp
    return PostData

if __name__=='__main__':
    model=keras.models.load_model('ScaleFactor2TrainIteration60k.h5')

    ScaleFactor=2
    HR_patch_size=(64,64)
    HR_slidingDis=8
    LR_patch_size=(32,32)
    LR_slidingDis=4

    
    HR_Image=np.load('./OverThrustClip.npy')

    LR_Image=np.load('10Hz_Result.npy')
    LR_Image=LR_Image.reshape((140,-1))
    LR_Image=LR_Image[26:-20,20:-20]
    
    Init_LR_Image=LR_Image
    Bucib_Image=skimage.transform.resize(LR_Image,output_shape=HR_Image.shape,mode='symmetric')
    
    LR_Image_Patch_,Patch_Idx=GetPatch(LR_Image,LR_patch_size,LR_slidingDis)
    HR_Image_Patch_,Patch_Idx=GetPatch(HR_Image,HR_patch_size,HR_slidingDis)
    
    LR_Image_Patch=Patch2TrainData(LR_Image_Patch_,LR_patch_size)
    HR_Image_Patch=Patch2TrainData(HR_Image_Patch_,HR_patch_size)
    
    MaxDataInput,MinDataInput,LR_Image_Patch=PreProcess_Data(LR_Image_Patch)
    MaxDataOutput,MinDataOutput,HR_Image_Patch=PreProcess_Data(HR_Image_Patch)
    
    Predict_Patch=model.predict(LR_Image_Patch)
    x_decoded=Predict_Patch[:,:,:,0]
    x_decoded=PostProcess_Data(x_decoded,MaxDataOutput,MinDataOutput)
    
    rows,cols=my_Im2col.ind2sub(np.array(HR_Image.shape)-HR_patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(HR_Image.shape)
    Weight=np.zeros(HR_Image.shape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=block
        Weight[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=np.ones(HR_patch_size)
        count+=1
    PredictData=IMout/Weight



    LR=skimage.transform.resize(Init_LR_Image,output_shape=HR_Image.shape,mode='symmetric')
    pyplot.figure(1)
    pyplot.imshow(Init_LR_Image)
    pyplot.figure(2)
    pyplot.imshow(LR)
    pyplot.figure(3)
    pyplot.imshow(PredictData)
    
