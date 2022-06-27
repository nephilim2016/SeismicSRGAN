#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:58:23 2021

@author: nephilim
"""

import tensorflow as tf
import numpy as np
# from numba import jit
import skimage.filters

class DeblurCNN(tf.keras.Model):
    def __init__(self):
        super(DeblurCNN,self).__init__()
        self.FirstLayer=tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')
        self.Conv2D=tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')
        self.bn=tf.keras.layers.BatchNormalization()
        self.LastLayer=tf.keras.layers.Conv2D(filters=1,kernel_size=3,strides=1,padding='same')
    def call(self,inputs,training=None):
        x=tf.nn.relu(self.FirstLayer(inputs))
        x_=x
        
        for idx in range(18):
            x=tf.nn.relu(self.bn(self.Conv2D(x),training=training))            
            x=tf.keras.layers.add([x_,x])
            x_=x
        x=tf.nn.sigmoid(self.LastLayer(x))
        
        return x
        
def MSE_loss_fn(Real,Pred):
    loss=tf.reduce_mean(tf.keras.losses.mean_squared_error(Real,Pred))
    return loss

# @jit(nopython=True)
def Create_Blur_Image(real_image):
    blur_image=np.zeros_like(real_image)
    for idx in range(blur_image.shape[0]):
        blur_image[idx,:,:,0]=skimage.filters.gaussian(real_image[idx,:,:,0],sigma=2)
    return blur_image

if __name__=='__main__':
    (Real_Image,_),(_,_)=tf.keras.datasets.mnist.load_data()
    Real_Image=Real_Image/255
    Real_Image=Real_Image.reshape(-1,28,28,1)
    Blur_Image=Create_Blur_Image(Real_Image)
    Real_Image=tf.data.Dataset.from_tensor_slices(Real_Image)
    Real_Image=Real_Image.batch(64)
    Blur_Image=tf.data.Dataset.from_tensor_slices(Blur_Image)
    Blur_Image=Blur_Image.batch(64)
    
    batch_size=64
    DeblurModel=DeblurCNN()
    DeblurModel.build(input_shape=(batch_size,28,28,1))
    DeblurModel.summary()
    
    Deblur_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.9)
    
    epoches=10
    deblur_losses=[]
    for epoch in range(epoches):
        for step,(real_image,blur_image) in enumerate(zip(Real_Image,Blur_Image)):
            with tf.GradientTape() as tape:
                Deblur_image=DeblurModel(blur_image)
                loss=MSE_loss_fn(real_image,Deblur_image)
            grads=tape.gradient(loss,DeblurModel.trainable_variables)
            Deblur_optimizer.apply_gradients(zip(grads,DeblurModel.trainable_variables))

            if step%100==0:
                print(step,'loss:',float(loss))

            deblur_losses.append(float(loss))

    DeblurModel.save_weights('DeblurModel.ckpt')








