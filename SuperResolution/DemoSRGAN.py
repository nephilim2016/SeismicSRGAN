#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:02:11 2020

@author: nephilim
"""

import SRGAN
import my_Im2col
import DataNormalized
import math
import numpy as np
import skimage.transform
from matplotlib import pyplot,cm
import os 


def PSNR(OriginalImage,BlurredImage):
    mse=np.sum((OriginalImage-BlurredImage)**2)/OriginalImage.size
    PSNR_=10*math.log10(np.max(OriginalImage)**2/mse)
    return PSNR_

def Patch2TrainData(Patch,PatchSize):
    TrainData=np.zeros(((Patch.shape[1],)+PatchSize+(1,)))
    for idx in range(Patch.shape[1]):
        TrainData[idx,:,:,0]=Patch[:,idx].reshape(PatchSize)
    return TrainData

def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def LoadData(FilePath,FileName,ScaleFactor):
    if isinstance(FileName,list):
        HR_Image_List=list()
        LR_Image_List=list()
        for filename in FileName:
            HR_Image=np.load(FilePath+'/'+filename)
            LR_Image=HR_Image[::ScaleFactor,::ScaleFactor]
            HR_Image_List.append(HR_Image)
            LR_Image_List.append(LR_Image)
        HR_patch_size=(64,64)
        HR_slidingDis=32
        LR_patch_size=(32,32)
        LR_slidingDis=16

        for idx in range(len(FileName)):
            HR_Image_Patch_,_=GetPatch(HR_Image_List[idx],HR_patch_size,HR_slidingDis)
            LR_Image_Patch_,_=GetPatch(LR_Image_List[idx],LR_patch_size,LR_slidingDis)
            if idx==0:
                LR_Image_Patch=LR_Image_Patch_
                HR_Image_Patch=HR_Image_Patch_
            else:
                LR_Image_Patch=np.hstack((LR_Image_Patch,LR_Image_Patch_))
                HR_Image_Patch=np.hstack((HR_Image_Patch,HR_Image_Patch_))
            
    else:
        HR_Image=np.load(FilePath+'/'+filename)
        LR_Image=HR_Image[::ScaleFactor,::ScaleFactor]
        HR_patch_size=(64,64)
        HR_slidingDis=32
        LR_patch_size=(32,32)
        LR_slidingDis=16
        HR_Image_Patch,_=GetPatch(HR_Image,HR_patch_size,HR_slidingDis)
        LR_Image_Patch,_=GetPatch(LR_Image,LR_patch_size,LR_slidingDis)
    TrainData_inputs=Patch2TrainData(LR_Image_Patch,LR_patch_size)
    TrainData_outputs=Patch2TrainData(HR_Image_Patch,HR_patch_size)
    return TrainData_inputs,TrainData_outputs
    
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

if __name__=='__main__':
    np.random.seed(10)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    FilePath='./SeismicResize'
    FileName=['BP94.npy','BP1994.npy','SEAM.npy','Marmousi.npy']
    ScaleFactor=2
    TrainData_inputs,TrainData_outputs=LoadData(FilePath,FileName,ScaleFactor)
    
    MaxData_Inputs,MinData_Inputs,TrainData_inputs=PreProcess_Data(TrainData_inputs)
    MaxData_Outputs,MinData_Outputs,TrainData_outputs=PreProcess_Data(TrainData_outputs)
    
    epochs=60000
    SRGAN=SRGAN.SRGAN(generator_image_shape=(32,32,1),discriminator_image_shape=(64,64,1),scale_factor=ScaleFactor)
    SRGAN_model=SRGAN.Build_GAN()
    
    HR_patch_size=(64,64)
    HR_slidingDis=32
    LR_patch_size=(32,32)
    LR_slidingDis=16
    inputs_train=TrainData_inputs
    outputs_train=TrainData_outputs
    save_path_name='./ScaleFactor2TrainIteration60k'
    generator,discriminator,adversarial=SRGAN.Train_GAN(TrainData_inputs,TrainData_outputs,save_path_name,epochs,batch_size=64)
    
    test=TrainData_inputs[20,:,:,0].reshape((1,32,32,1))
    fake_image=generator(test)
    fake_image=fake_image.numpy()
    fake_image=fake_image[0,:,:,0]
    lr_image=test[0,:,:,0]
    real_image=TrainData_outputs[20,:,:,0]