#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:03:55 2021

@author: nephilim
"""

import tensorflow as tf
import numpy as np
import skimage.filters
from numba import jit
import my_Im2col
import DataNormalized
import math
from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def Create_Blur_Image(real_image):
    blur_image=np.zeros_like(real_image)
    for idx in range(blur_image.shape[0]):
        blur_image[idx,:,:,0]=skimage.filters.gaussian(real_image[idx,:,:,0],sigma=4)
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
        slidingDis=8
        Real_Image_Patch,_=GetPatch(Real_Image_List[idx],patch_size,slidingDis)
    TrainData_outputs=Patch2TrainData(Real_Image_Patch,patch_size)
    return TrainData_outputs

def PostProcess_Data(Data,MaxData,MinData):
    PostData=np.zeros_like(Data)
    for idx in range(Data.shape[0]):
        Data_tmp=DataNormalized.InverseDataNormalized(Data[idx,:,:],MaxData[idx],MinData[idx])
        PostData[idx,:,:]=Data_tmp
    return PostData


if __name__=='__main__':
    model=tf.keras.models.load_model('SeismicDeblurModel500.h5',custom_objects={'MSE_loss_fn':MSE_loss_fn})
    
    Clean_Image=np.load('./SeismicResize/OverThrustClip.npy')
    # Clean_Image=Clean_Image[::4,::4]
    Blur_Image=np.load('10Hz_LR.npy')
    Blur_Image=Blur_Image.reshape((140,-1))
    Blur_Image=Blur_Image[26:-20,20:-20]
    
    # sigma=5
    # Blur_Image=skimage.filters.gaussian(Clean_Image,sigma)
    patch_size=(64,64)
    slidingDis=8
    
    Clean_Image_Patch_,Patch_Idx=GetPatch(Clean_Image,patch_size,slidingDis)
    Blur_Image_Patch_,Patch_Idx=GetPatch(Blur_Image,patch_size,slidingDis)
    
    Clean_Image_Patch=Patch2TrainData(Clean_Image_Patch_,patch_size)
    Blur_Image_Patch=Patch2TrainData(Blur_Image_Patch_,patch_size)
    
    MaxData_Inputs,MinData_Inputs,Clean_Image_Patch=PreProcess_Data(Clean_Image_Patch)
    MaxData_Outputs,MinData_Outputs,Blur_Image_Patch=PreProcess_Data(Blur_Image_Patch)
    
    Predict_Patch=model.predict(Blur_Image_Patch)
    Predict_Patch=model.predict(Predict_Patch)
    Predict_Patch=model.predict(Predict_Patch)
    x_decoded=Predict_Patch[:,:,:,0]
    x_decoded=PostProcess_Data(x_decoded,MaxData_Outputs,MinData_Outputs)
    
    rows,cols=my_Im2col.ind2sub(np.array(Clean_Image.shape)-patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(Clean_Image.shape)
    Weight=np.zeros(Clean_Image.shape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+patch_size[0],col:col+patch_size[1]]+=block
        Weight[row:row+patch_size[0],col:col+patch_size[1]]+=np.ones(patch_size)
        count+=1
    PredictData=IMout/Weight
    
    pyplot.figure(1)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1880,0]
    gci=pyplot.imshow(Blur_Image,cmap=cm.seismic,extent=extent,norm=norm)
    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='5%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    # pyplot.savefig('sigma_%s_Result.png'%sigma,dpi=1000)
    
    pyplot.figure(2)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1880,0]
    gci=pyplot.imshow(PredictData,cmap=cm.seismic,extent=extent,norm=norm)
    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='5%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    # pyplot.savefig('sigma_%s_Deblur_Result.png'%sigma,dpi=1000)
    
    # pyplot.figure(3)
    # norm=colors.Normalize(vmin=2500, vmax=6000)
    # extent=[0,4000,1880,0]
    # gci=pyplot.imshow(Clean_Image,cmap=cm.seismic,extent=extent,norm=norm)
    # ax=pyplot.gca()
    # divider=make_axes_locatable(ax)
    # cax=divider.append_axes('right', size='5%', pad=0.15)
    # cbar=pyplot.colorbar(gci,cax=cax)
    # cbar.set_label('$m/s$')
    # ax.set_xlabel('Position (m)')
    # ax.set_ylabel('Depth (m)') 
    # pyplot.savefig('Clean_Result.png',dpi=1000)
    pyplot.figure(3)
    index=190
    vp_true_line=Clean_Image[:,index]
    vp_blur_line=Blur_Image[:,index]
    vp_deblur_line=PredictData[:,index]
    pyplot.plot(vp_true_line,np.linspace(0,1870,187),'k--')
    pyplot.plot(vp_blur_line,np.linspace(0,1870,187),'b-.')
    pyplot.plot(vp_deblur_line,np.linspace(0,1870,187),'r-')
    ax=pyplot.gca()
    ax.invert_yaxis()
    ax.set(aspect=3)
    pyplot.legend(['True Model','Low Resolution','SRE-ResNet'],frameon=False)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Depth (m)') 
    # pyplot.savefig('sigma_%s_Line_Result.png'%sigma,dpi=1000)
    
    