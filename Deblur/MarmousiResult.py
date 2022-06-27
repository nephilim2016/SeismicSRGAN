#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:37:24 2021

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
    
    Freq_5Hz_Image=np.load('./5Hz_Marmousi.npy')
    Freq_5Hz_Image=Freq_5Hz_Image.reshape((115,-1))
    Freq_5Hz_Image=Freq_5Hz_Image[20:-20,20:-20]
    # Freq_5Hz_Image=Freq_5Hz_Image[::4,::4]
    Freq_5Hz_Blur_Image=np.load('./5Hz_Marmousi.npy')
    Freq_5Hz_Blur_Image=Freq_5Hz_Blur_Image.reshape((115,-1))
    Freq_5Hz_Blur_Image=Freq_5Hz_Blur_Image[20:-20,20:-20]
    
    # sigma=5
    # Freq_5Hz_Blur_Image=skimage.filters.gaussian(Freq_5Hz_Image,sigma)
    patch_size=(64,64)
    slidingDis=4
    
    Freq_5Hz_Image_Patch_,Patch_Idx=GetPatch(Freq_5Hz_Image,patch_size,slidingDis)
    Freq_5Hz_Blur_Image_Patch_,Patch_Idx=GetPatch(Freq_5Hz_Blur_Image,patch_size,slidingDis)
    
    Freq_5Hz_Image_Patch=Patch2TrainData(Freq_5Hz_Image_Patch_,patch_size)
    Freq_5Hz_Blur_Image_Patch=Patch2TrainData(Freq_5Hz_Blur_Image_Patch_,patch_size)
    
    MaxData_Inputs,MinData_Inputs,Freq_5Hz_Image_Patch=PreProcess_Data(Freq_5Hz_Image_Patch)
    MaxData_Outputs,MinData_Outputs,Freq_5Hz_Blur_Image_Patch=PreProcess_Data(Freq_5Hz_Blur_Image_Patch)
    
    Predict_Patch=model.predict(Freq_5Hz_Blur_Image_Patch)
    Predict_Patch=model.predict(Predict_Patch)
    Predict_Patch=model.predict(Predict_Patch)
    x_decoded=Predict_Patch[:,:,:,0]
    x_decoded=PostProcess_Data(x_decoded,MaxData_Outputs,MinData_Outputs)
    
    rows,cols=my_Im2col.ind2sub(np.array(Freq_5Hz_Image.shape)-patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(Freq_5Hz_Image.shape)
    Weight=np.zeros(Freq_5Hz_Image.shape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+patch_size[0],col:col+patch_size[1]]+=block
        Weight[row:row+patch_size[0],col:col+patch_size[1]]+=np.ones(patch_size)
        count+=1
    Freq_5Hz_Deblur_Image=IMout/Weight
    ###########
    Freq_5Hz_Deblur_Image=denoising_2D_TV(Freq_5Hz_Deblur_Image)
    pyplot.figure()
    norm=colors.Normalize(vmin=1400, vmax=6000)
    extent=[0,10000,3000,0]
    gci=pyplot.imshow(Freq_5Hz_Deblur_Image,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,10000,10)
    y=np.linspace(0,3000,50)
    for idx_x in idx:
        if np.mod(idx_x,40)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)

        
    idx=np.arange(0,3000,10)
    x=np.linspace(0,10000,50)
    for idx_z in idx:
        if np.mod(idx_z,40)==0:
            pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)

    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='3%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    
    
    ##############################################################################################
    Freq_10Hz_Image=np.load('./10Hz_Marmousi.npy')
    Freq_10Hz_Image=Freq_10Hz_Image.reshape((190,-1))
    Freq_10Hz_Image=Freq_10Hz_Image[20:-20,20:-20]
    # Freq_5Hz_Image=Freq_5Hz_Image[::4,::4]
    Freq_10Hz_Blur_Image=np.load('./10Hz_Marmousi.npy')
    Freq_10Hz_Blur_Image=Freq_10Hz_Blur_Image.reshape((190,-1))
    Freq_10Hz_Blur_Image=Freq_10Hz_Blur_Image[20:-20,20:-20]
    
    # sigma=5
    # Freq_5Hz_Blur_Image=skimage.filters.gaussian(Freq_5Hz_Image,sigma)
    patch_size=(64,64)
    slidingDis=4
    
    Freq_10Hz_Image_Patch_,Patch_Idx=GetPatch(Freq_10Hz_Image,patch_size,slidingDis)
    Freq_10Hz_Blur_Image_Patch_,Patch_Idx=GetPatch(Freq_10Hz_Blur_Image,patch_size,slidingDis)
    
    Freq_10Hz_Image_Patch=Patch2TrainData(Freq_10Hz_Image_Patch_,patch_size)
    Freq_10Hz_Blur_Image_Patch=Patch2TrainData(Freq_10Hz_Blur_Image_Patch_,patch_size)
    
    MaxData_Inputs,MinData_Inputs,Freq_10Hz_Image_Patch=PreProcess_Data(Freq_10Hz_Image_Patch)
    MaxData_Outputs,MinData_Outputs,Freq_10Hz_Blur_Image_Patch=PreProcess_Data(Freq_10Hz_Blur_Image_Patch)
    
    Predict_Patch=model.predict(Freq_10Hz_Blur_Image_Patch)
    Predict_Patch=model.predict(Predict_Patch)
    Predict_Patch=model.predict(Predict_Patch)
    x_decoded=Predict_Patch[:,:,:,0]
    x_decoded=PostProcess_Data(x_decoded,MaxData_Outputs,MinData_Outputs)
    
    rows,cols=my_Im2col.ind2sub(np.array(Freq_10Hz_Image.shape)-patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(Freq_10Hz_Image.shape)
    Weight=np.zeros(Freq_10Hz_Image.shape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+patch_size[0],col:col+patch_size[1]]+=block
        Weight[row:row+patch_size[0],col:col+patch_size[1]]+=np.ones(patch_size)
        count+=1
    Freq_10Hz_Deblur_Image=IMout/Weight
    ###########
    Freq_10Hz_Deblur_Image=denoising_2D_TV(Freq_10Hz_Deblur_Image)
    pyplot.figure()
    norm=colors.Normalize(vmin=1400, vmax=6000)
    extent=[0,10000,3000,0]
    gci=pyplot.imshow(Freq_10Hz_Deblur_Image,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,10000,10)
    y=np.linspace(0,3000,50)
    for idx_x in idx:
        if np.mod(idx_x,40)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        elif np.mod(idx_x,40)==20:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
    idx=np.arange(0,3000,10)
    x=np.linspace(0,10000,50)
    for idx_z in idx:
        if np.mod(idx_z,40)==0:
            pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        elif np.mod(idx_z,40)==20:
            pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)

    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='3%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    
    
    ##############################################################################################
    Freq_20Hz_Image=np.load('./20Hz_Marmousi.npy')
    Freq_20Hz_Image=Freq_20Hz_Image.reshape((340,-1))
    Freq_20Hz_Image=Freq_20Hz_Image[20:-20,20:-20]
    # Freq_5Hz_Image=Freq_5Hz_Image[::4,::4]
    Freq_20Hz_Blur_Image=np.load('./20Hz_Marmousi.npy')
    Freq_20Hz_Blur_Image=Freq_20Hz_Blur_Image.reshape((340,-1))
    Freq_20Hz_Blur_Image=Freq_20Hz_Blur_Image[20:-20,20:-20]
    
    # sigma=5
    # Freq_5Hz_Blur_Image=skimage.filters.gaussian(Freq_5Hz_Image,sigma)
    patch_size=(64,64)
    slidingDis=4
    
    Freq_20Hz_Image_Patch_,Patch_Idx=GetPatch(Freq_20Hz_Image,patch_size,slidingDis)
    Freq_20Hz_Blur_Image_Patch_,Patch_Idx=GetPatch(Freq_20Hz_Blur_Image,patch_size,slidingDis)
    
    Freq_20Hz_Image_Patch=Patch2TrainData(Freq_20Hz_Image_Patch_,patch_size)
    Freq_20Hz_Blur_Image_Patch=Patch2TrainData(Freq_20Hz_Blur_Image_Patch_,patch_size)
    
    MaxData_Inputs,MinData_Inputs,Freq_20Hz_Image_Patch=PreProcess_Data(Freq_20Hz_Image_Patch)
    MaxData_Outputs,MinData_Outputs,Freq_20Hz_Blur_Image_Patch=PreProcess_Data(Freq_20Hz_Blur_Image_Patch)
    
    Predict_Patch=model.predict(Freq_20Hz_Blur_Image_Patch)
    Predict_Patch=model.predict(Predict_Patch)
    Predict_Patch=model.predict(Predict_Patch)
    x_decoded=Predict_Patch[:,:,:,0]
    x_decoded=PostProcess_Data(x_decoded,MaxData_Outputs,MinData_Outputs)
    
    rows,cols=my_Im2col.ind2sub(np.array(Freq_20Hz_Image.shape)-patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(Freq_20Hz_Image.shape)
    Weight=np.zeros(Freq_20Hz_Image.shape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+patch_size[0],col:col+patch_size[1]]+=block
        Weight[row:row+patch_size[0],col:col+patch_size[1]]+=np.ones(patch_size)
        count+=1
    Freq_20Hz_Deblur_Image=IMout/Weight
    ###########
    Freq_20Hz_Deblur_Image=denoising_2D_TV(Freq_20Hz_Deblur_Image)
    pyplot.figure()
    norm=colors.Normalize(vmin=1400, vmax=6000)
    extent=[0,10000,3000,0]
    gci=pyplot.imshow(Freq_20Hz_Deblur_Image,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,10000,10)
    y=np.linspace(0,3000,50)
    for idx_x in idx:
        if np.mod(idx_x,40)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        elif np.mod(idx_x,10)==0:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
    idx=np.arange(0,3000,10)
    x=np.linspace(0,10000,50)
    for idx_z in idx:
        if np.mod(idx_z,40)==0:
            pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        elif np.mod(idx_z,10)==0:
            pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)

    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='3%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    
    
    
    
    Real_Marmousi=np.load('Marmousi.npy')*1333
    
    pyplot.figure()
    norm=colors.Normalize(vmin=1400, vmax=6000)
    extent=[0,10000,3000,0]
    gci=pyplot.imshow(Real_Marmousi,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,10000,10)
    y=np.linspace(0,3000,50)
    for idx_x in idx:
        if np.mod(idx_x,40)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        elif np.mod(idx_x,10)==0:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
    idx=np.arange(0,3000,10)
    x=np.linspace(0,10000,50)
    for idx_z in idx:
        if np.mod(idx_z,40)==0:
            pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        elif np.mod(idx_z,10)==0:
            pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)

    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='3%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    
    Real_Marmousi=skimage.transform.resize(Real_Marmousi,output_shape=(300,1000),mode='symmetric')
    
    x_10=np.arange(0,3000,10)
    x_20=np.arange(0,3000,20)
    x_40=np.arange(0,3000,40)
    
    pyplot.plot(Real_Marmousi[:,600],x_10,'k-')
    
    pyplot.plot(Freq_5Hz_Deblur_Image[:,150],x_40,'g-')
    pyplot.plot(Freq_10Hz_Deblur_Image[:,300],x_20,'b-.')
    pyplot.plot(Freq_20Hz_Deblur_Image[:,600],x_10,'r-')
    
    ax=pyplot.gca()
    ax.invert_yaxis()
    ax.set(aspect=3)
    
    pyplot.legend(['True Model','5Hz FWI+RE','10Hz FWI+RE','20Hz FWI+RE'],frameon=False)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Depth (m)') 
