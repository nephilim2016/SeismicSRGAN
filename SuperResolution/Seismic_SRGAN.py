#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:55:18 2021

@author: nephilim
"""

import tensorflow as tf
import numpy as np
from numba import jit
import my_Im2col
import DataNormalized
import math
from matplotlib import pyplot,cm
import os 

# Create the discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        
        self.Conv0=tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')
        
        filters=64
        self.Conv1=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=2,padding='same')
        self.bn1=tf.keras.layers.BatchNormalization()
        
        self.Conv2=tf.keras.layers.Conv2D(filters=2*filters,kernel_size=3,strides=1,padding='same')
        self.bn2=tf.keras.layers.BatchNormalization()
        self.Conv3=tf.keras.layers.Conv2D(filters=2*filters,kernel_size=3,strides=2,padding='same')
        self.bn3=tf.keras.layers.BatchNormalization()
        
        self.Conv4=tf.keras.layers.Conv2D(filters=4*filters,kernel_size=3,strides=1,padding='same')
        self.bn4=tf.keras.layers.BatchNormalization()
        self.Conv5=tf.keras.layers.Conv2D(filters=4*filters,kernel_size=3,strides=2,padding='same')
        self.bn5=tf.keras.layers.BatchNormalization()
        
        # self.Conv6=tf.keras.layers.Conv2D(filters=8*filters,kernel_size=3,strides=1,padding='same')
        # self.bn6=tf.keras.layers.BatchNormalization()
        # self.Conv7=tf.keras.layers.Conv2D(filters=8*filters,kernel_size=3,strides=2,padding='same')
        # self.bn7=tf.keras.layers.BatchNormalization()
        
        self.flatten=tf.keras.layers.Flatten()
        
        # self.dense0=tf.keras.layers.Dense(units=1024)
        
        self.dense1=tf.keras.layers.Dense(units=1)
    
    def call(self,inputs,training=None):
        x=tf.nn.leaky_relu(self.Conv0(inputs))
        
        x=tf.nn.leaky_relu(self.bn1(self.Conv1(x),training=training))
        x=tf.nn.leaky_relu(self.bn2(self.Conv2(x),training=training))
        x=tf.nn.leaky_relu(self.bn3(self.Conv3(x),training=training))
        x=tf.nn.leaky_relu(self.bn4(self.Conv4(x),training=training))
        x=tf.nn.leaky_relu(self.bn5(self.Conv5(x),training=training))
        # x=tf.nn.leaky_relu(self.bn6(self.Conv6(x),training=training))
        # x=tf.nn.leaky_relu(self.bn7(self.Conv7(x),training=training))
        x=self.flatten(x)
        # x=tf.nn.leaky_relu(self.dense0(x))
        x=self.dense1(x)
        
        return x

# Create the discriminator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        
        self.Conv0=tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')
        
        filters=64
        self.Conv10=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn10=tf.keras.layers.BatchNormalization()
        self.Conv11=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn11=tf.keras.layers.BatchNormalization()
        
        self.Conv20=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn20=tf.keras.layers.BatchNormalization()
        self.Conv21=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn21=tf.keras.layers.BatchNormalization()
        
        self.Conv30=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn30=tf.keras.layers.BatchNormalization()
        self.Conv31=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn31=tf.keras.layers.BatchNormalization()
        
        self.Conv40=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn40=tf.keras.layers.BatchNormalization()
        self.Conv41=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn41=tf.keras.layers.BatchNormalization()
        
        self.Conv50=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn50=tf.keras.layers.BatchNormalization()
        self.Conv51=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn51=tf.keras.layers.BatchNormalization()
        
        self.Conv6=tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=1,padding='same')
        self.bn6=tf.keras.layers.BatchNormalization()
        
        self.PixelShuffler=tf.keras.layers.Conv2D(filters=4*filters,kernel_size=3,strides=1,padding='same')
        
        self.Conv7=tf.keras.layers.Conv2D(filters=1,kernel_size=9,strides=1,padding='same')
    
    
    
    def call(self,inputs,training=None):
        x_layer=tf.nn.leaky_relu(self.Conv0(inputs))
        
        x=tf.nn.leaky_relu(self.bn10(self.Conv10(x_layer),training=training))
        x=self.bn11(self.Conv11(x),training=training)
        x_res_0=tf.keras.layers.add([x_layer,x])
        
        
        x=tf.nn.leaky_relu(self.bn20(self.Conv20(x_res_0),training=training))
        x=self.bn21(self.Conv21(x),training=training)
        x_res_1=tf.keras.layers.add([x_res_0,x])
        
        x=tf.nn.leaky_relu(self.bn30(self.Conv30(x_res_1),training=training))
        x=self.bn31(self.Conv31(x),training=training)
        x_res_2=tf.keras.layers.add([x_res_1,x])
        
        x=tf.nn.leaky_relu(self.bn40(self.Conv40(x_res_2),training=training))
        x=self.bn41(self.Conv41(x),training=training)
        x_res_3=tf.keras.layers.add([x_res_2,x])
        
        x=tf.nn.leaky_relu(self.bn50(self.Conv50(x_res_3),training=training))
        x=self.bn51(self.Conv51(x),training=training)
        x_res_4=tf.keras.layers.add([x_res_3,x])
        
        
        x=self.bn6(self.Conv6(x_res_4),training=training)
        x=tf.keras.layers.add([x_layer,x])
        
        x=self.PixelShuffler(x)
        x=tf.nn.depth_to_space(x,2)
        
        x=self.Conv7(x)
        x=tf.nn.tanh(x)
        
        return x     


def celoss_ones(logits):
    y=tf.ones_like(logits)
    loss=tf.keras.losses.binary_crossentropy(y,logits,from_logits=True)
    return tf.reduce_mean(loss)

def celoss_zeros(logits):
    y=tf.zeros_like(logits)
    loss=tf.keras.losses.binary_crossentropy(y,logits,from_logits=True)
    return tf.reduce_mean(loss)

def d_loss_fn(generator,discriminator,LR_image,Real_image,is_training):
    fake_image=generator(LR_image,is_training)
    d_fake_logits=discriminator(fake_image,is_training)
    d_real_logits=discriminator(Real_image,is_training)
    d_loss_real=celoss_ones(d_real_logits)
    d_loss_fake=celoss_zeros(d_fake_logits)
    loss=d_loss_real+d_loss_fake
    return loss        

def g_loss_fn(generator,discriminator,LR_image,is_training):
    fake_image=generator(LR_image,is_training)
    d_fake_logits=discriminator(fake_image,is_training)
    loss=celoss_ones(d_fake_logits)
    return loss         

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
        HR_patch_size=(128,128)
        HR_slidingDis=32
        LR_patch_size=(64,64)
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
        HR_patch_size=(128,128)
        HR_slidingDis=32
        LR_patch_size=(64,644)
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
    FilePath='./SeismicResize'
    FileName=['BP94.npy','BP1994.npy','SEAM.npy','Marmousi.npy']
    ScaleFactor=2
    TrainData_inputs,TrainData_outputs=LoadData(FilePath,FileName,ScaleFactor)
    
    MaxData_Inputs,MinData_Inputs,TrainData_inputs=PreProcess_Data(TrainData_inputs)
    MaxData_Outputs,MinData_Outputs,TrainData_outputs=PreProcess_Data(TrainData_outputs)
    
    batch_size=32
    generator=Generator()
    generator.build(input_shape=(batch_size,64,64,1))
    generator.summary()
    discriminator=Discriminator()
    discriminator.build(input_shape=(batch_size,128,128,1))
    discriminator.summary()
    
    # lr=2e-4
    # decay=6e-8
    # g_optimizer=tf.keras.optimizers.RMSprop(lr=0.5*lr,decay=0.5*decay)
    # d_optimizer=tf.keras.optimizers.RMSprop(lr=lr,decay=decay)
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.9)
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.9)
    
    train_size=TrainData_inputs.shape[0]
    epoches=10000
    d_losses=[]
    g_losses=[]
    for epoch in range(epoches):
        for _ in range(2):
            rand_indexes=np.random.randint(0,train_size,size=batch_size)
            LR_image=TrainData_inputs[rand_indexes]
            Real_image=TrainData_outputs[rand_indexes]
            with tf.GradientTape() as tape:
                d_loss=d_loss_fn(generator,discriminator,LR_image,Real_image,is_training=True)
            grads=tape.gradient(d_loss,discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads,discriminator.trainable_variables))
        
        rand_indexes=np.random.randint(0,train_size,size=batch_size)
        LR_image=TrainData_inputs[rand_indexes]
        Real_image=TrainData_outputs[rand_indexes]
        with tf.GradientTape() as tape:
            g_loss=g_loss_fn(generator,discriminator,LR_image,is_training=True)
        grads=tape.gradient(g_loss,generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads,generator.trainable_variables))
        
        if epoch%1==0:
            print(epoch, 'd-loss:',float(d_loss), 'g-loss:', float(g_loss))

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

    generator.save_weights('generator.ckpt')
    discriminator.save_weights('discriminator.ckpt')
    
    

    test=TrainData_inputs[0,:,:,0].reshape((1,64,64,1))
    fake_image=generator(test)
    fake_image=fake_image.numpy()
    fake_image=fake_image[0,:,:,0]
    lr_image=test[0,:,:,0]
    real_image=TrainData_outputs[0,:,:,0]
