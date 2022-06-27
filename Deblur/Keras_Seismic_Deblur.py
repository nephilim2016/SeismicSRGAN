#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 23:12:33 2021

@author: nephilim
"""


import tensorflow as tf
import numpy as np
import skimage.filters
from numba import jit
import my_Im2col
import DataNormalized
import math
from matplotlib import pyplot,cm
import os 

class SeismicDeblur():
    def __init__(self,image_shape):
        self.image_shape=image_shape
    
    def Build_Deblur(self):
        
        def ResidualBlock(layer_input,filters,kernel_size):
            x=tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(layer_input)
            x=tf.keras.layers.BatchNormalization()(x)
            x=tf.keras.layers.Activation('relu')(x)
            x=tf.keras.layers.add([layer_input,x])
            return x
        
        input_images=tf.keras.layers.Input(shape=self.image_shape)
        x=tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')(input_images)
        x_FirstLayer=tf.keras.layers.Activation('relu')(x)
        x=x_FirstLayer
        for _ in range(8):
            x=ResidualBlock(x,64,3)
        x=tf.keras.layers.add([x,x_FirstLayer])
        x=tf.keras.layers.Conv2D(filters=1,kernel_size=3,strides=1,padding='same')(x)
        x=tf.keras.layers.Activation('sigmoid')(x)
        
        model=tf.keras.models.Model(inputs=input_images,outputs=x)
        model.summary()
        
        return model
    
def MSE_loss_fn(Real,Pred):
    loss=tf.keras.backend.mean(tf.keras.losses.mean_squared_error(Real,Pred))
    return loss

def BuildDeblur(ImageShape=(64,64,1)):
    SeismicDeblur_=SeismicDeblur(ImageShape)
    Model=SeismicDeblur_.Build_Deblur()
    Model.summary()  
    optimizer=tf.keras.optimizers.Adam(lr=0.0002,beta_1=0.9,beta_2=0.999)
    Model.compile(loss=MSE_loss_fn,optimizer=optimizer)
    return Model

def DeblurTraining(Model,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name):
    callbacks_list=[tf.keras.callbacks.ModelCheckpoint(filepath=save_path_name+'.h5',monitor='val_loss',save_best_only=True),\
                    tf.keras.callbacks.TensorBoard(log_dir='./TensorBoard',histogram_freq=1,write_graph=True,write_images=True)]
    history=Model.fit(inputs_train,outputs_train,epochs=epochs,batch_size=64,callbacks=callbacks_list,validation_data=(inputs_validation,outputs_validation))
    test_loss=Model.evaluate(inputs_validation,outputs_validation)
    return history,test_loss,Model

def Create_Blur_Image(real_image,sigma):
    blur_image=np.zeros_like(real_image)
    for idx in range(blur_image.shape[0]):
        blur_image[idx,:,:,0]=skimage.filters.gaussian(real_image[idx,:,:,0],sigma=1)
    return blur_image

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

def LoadData(FilePath,FileName):
    if isinstance(FileName,list):
        Real_Image_List=list()
        for filename in FileName:
            Real_Image=np.load(FilePath+'/'+filename)
            Real_Image_List.append(Real_Image)
        patch_size=(64,64)
        slidingDis=8

        for idx in range(len(FileName)):
            Real_Image_Patch_,_=GetPatch(Real_Image_List[idx],patch_size,slidingDis)
            if idx==0:
                Real_Image_Patch=Real_Image_Patch_
            else:
                Real_Image_Patch=np.hstack((Real_Image_Patch,Real_Image_Patch_))
            
    else:
        Real_Image=np.load(FilePath+'/'+filename)
        patch_size=(64,64)
        slidingDis=24
        Real_Image_Patch,_=GetPatch(Real_Image_List[idx],patch_size,slidingDis)
    TrainData_outputs=Patch2TrainData(Real_Image_Patch,patch_size)
    return TrainData_outputs

if __name__=='__main__':
    FilePath='./SeismicResize'
    FileName=['BP94.npy','BP1994.npy','SEAM.npy','Marmousi.npy']
    TrainData_outputs=LoadData(FilePath,FileName)
    TrainData_inputs=Create_Blur_Image(TrainData_outputs,sigma=1)
    
    MaxData_Inputs,MinData_Inputs,TrainData_inputs=PreProcess_Data(TrainData_inputs)
    MaxData_Outputs,MinData_Outputs,TrainData_outputs=PreProcess_Data(TrainData_outputs)
    
    batch_size=64
    ImageShape=(64,64,1)
    DeblurModel=BuildDeblur(ImageShape=ImageShape)
    epochs=500
    inputs_train=TrainData_inputs[:32000]
    outputs_train=TrainData_outputs[:32000]
    inputs_validation=TrainData_inputs[32000:]
    outputs_validation=TrainData_outputs[32000:]
    save_path_name='./SeismicDeblurModel'
    history,test_loss,Model=DeblurTraining(DeblurModel,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name)
    
    
    
    blur_image=inputs_train[0,:,:,0]
    blur_image_reshape=np.reshape(blur_image,[1,64,64,1])
    deblur=Model.predict(blur_image_reshape)
    deblur_image=deblur[0,:,:,0]
    real_image=outputs_train[0,:,:,0]
    