#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 20:11:31 2021

@author: nephilim
"""

import tensorflow as tf
import numpy as np
import skimage.filters

class MNISTDeblur():
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
        for _ in range(18):
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

def BuildDeblur(ImageShape=(28,28,1)):
    MNISTDeblur_=MNISTDeblur(ImageShape)
    Model=MNISTDeblur_.Build_Deblur()
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
        blur_image[idx,:,:,0]=skimage.filters.gaussian(real_image[idx,:,:,0],sigma=2)
    return blur_image

if __name__=='__main__':
    (Real_Image,_),(_,_)=tf.keras.datasets.mnist.load_data()
    Real_Image=Real_Image/255
    Real_Image=Real_Image.reshape(-1,28,28,1)
    Blur_Image=Create_Blur_Image(Real_Image)
    
    batch_size=64
    ImageShape=(28,28,1)
    DeblurModel=BuildDeblur(ImageShape=ImageShape)
    epochs=10
    inputs_train=Blur_Image[:50000]
    outputs_train=Real_Image[:50000]
    inputs_validation=Blur_Image[50000:]
    outputs_validation=Real_Image[50000:]
    save_path_name='./DeblurModel'
    history,test_loss,Model=DeblurTraining(DeblurModel,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name)
    
    
    
    blur_image=Blur_Image[0,:,:,0]
    blur_image_reshape=np.reshape(blur_image,[1,28,28,1])
    deblur=Model.predict(blur_image_reshape)
    deblur_image=deblur[0,:,:,0]
    real_image=Real_Image[0,:,:,0]
    