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

def LoadData(FilePath,FileName,ImageShape,ScaleFactor):
    if isinstance(FileName,list):
        HR_Image_List=list()
        LR_Image_List=list()
        for filename in FileName:
            data_temp=np.load(FilePath+'/'+filename)
            data_temp=skimage.transform.resize(data_temp,output_shape=ImageShape,mode='symmetric')
            HR_Image=DataNormalized.DataNormalized(data_temp)
            LR_Image=HR_Image[::ScaleFactor,::ScaleFactor]
            HR_Image_List.append(HR_Image)
            LR_Image_List.append(LR_Image)
        HR_patch_size=(28,28)
        HR_slidingDis=12
        LR_patch_size=(7,7)
        LR_slidingDis=3

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
        data_temp=np.load(FilePath+'/'+filename)
        data_temp=skimage.transform.resize(data_temp,output_shape=ImageShape,mode='symmetric')
        HR_Image=DataNormalized.DataNormalized(data_temp)
        LR_Image=HR_Image[::ScaleFactor,::ScaleFactor]
        HR_patch_size=(28,28)
        HR_slidingDis=12
        LR_patch_size=(7,7)
        LR_slidingDis=3
        HR_Image_Patch,_=GetPatch(HR_Image,HR_patch_size,HR_slidingDis)
        LR_Image_Patch,_=GetPatch(LR_Image,LR_patch_size,LR_slidingDis)
    TrainData_inputs=Patch2TrainData(LR_Image_Patch,LR_patch_size)
    TrainData_outputs=Patch2TrainData(HR_Image_Patch,HR_patch_size)
    return TrainData_inputs,TrainData_outputs
    
if __name__=='__main__':
    np.random.seed(10)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    FilePath='./SeismicModel'
    # FileName=['bp94Vp.npy','BP_1994_vp.npy','salt_x12.5_z6.25_rho.npy','SEAM_Vp.npy','vel_z6.25m_x12.5m_exact.npy','vel_z6.25m_x12.5m_nosalt.npy','vp_marmousi-ii.npy']
    FileName=['vp_marmousi-ii.npy','bp94Vp.npy']
    ImageShape=(200,400)
    ScaleFactor=4
    epochs=60000
    SRGAN=SRGAN.SRGAN(generator_image_shape=(7,7,1),discriminator_image_shape=(28,28,1),scale_factor=ScaleFactor)
    SRGAN_model=SRGAN.Build_GAN()
    
    HR_patch_size=(28,28)
    HR_slidingDis=12
    LR_patch_size=(7,7)
    LR_slidingDis=3
    TrainData_inputs,TrainData_outputs=LoadData(FilePath,FileName,ImageShape,ScaleFactor)
    inputs_train=TrainData_inputs
    outputs_train=TrainData_outputs
    save_path_name='./ScaleFactor4TrainIteration200'
    generator,discriminator,adversarial=SRGAN.Train_GAN(TrainData_inputs,TrainData_outputs,save_path_name,epochs,batch_size=8)
    
    HR_Image=np.load('./SeismicModel/bp94Vp.npy')
    HR_Image=skimage.transform.resize(HR_Image,output_shape=ImageShape,mode='symmetric')
    HR_Image=DataNormalized.DataNormalized(HR_Image)
    LR_Image=HR_Image[::ScaleFactor,::ScaleFactor]
    Bucib_Image=skimage.transform.resize(LR_Image,output_shape=ImageShape,mode='symmetric')
    LR_Image_Patch_,Patch_Idx=GetPatch(LR_Image,LR_patch_size,LR_slidingDis)
    HR_Image_Patch_,Patch_Idx=GetPatch(HR_Image,HR_patch_size,HR_slidingDis)
    TestData=Patch2TrainData(LR_Image_Patch_,LR_patch_size)
    Predict_Patch=generator.predict(TestData)
    x_decoded=Predict_Patch[:,:,:,0]
    rows,cols=my_Im2col.ind2sub(np.array(ImageShape)-HR_patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(ImageShape)
    Weight=np.zeros(ImageShape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=block
        Weight[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=np.ones(HR_patch_size)
        count+=1
    PredictData=IMout/Weight
    
    LR=skimage.transform.resize(TestData[20,:,:,0],output_shape=(28,28),mode='symmetric')
    HR=x_decoded[20,:,:]
    pyplot.figure(1)
    pyplot.imshow(LR)
    pyplot.figure(2)
    pyplot.imshow(HR)
    
    print(PSNR(HR_Image,Bucib_Image))
    print(PSNR(HR_Image,PredictData))
    
    
    
    LR_image=inputs_train[20,:,:,0]
    HR_image=outputs_train[20,:,:,0]
    input_test=inputs_train[20,:,:,:].reshape((1,7,7,1))
    generate_image=generator.predict(input_test)
    generate_image=generate_image[0,:,:,0]
    discriminator.predict(HR_image.reshape((1,28,28,1)))
    discriminator.predict(generate_image.reshape((1,28,28,1)))
    adversarial.predict(input_test)
    
    # pyplot.figure()
    # pyplot.imshow(inputs_train[20,:,:,0])
    # pyplot.figure()
    # pyplot.imshow(outputs_train[20,:,:,0])