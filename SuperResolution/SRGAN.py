#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:54:31 2020

@author: nephilim
"""

# SRGAN Reference by Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
import tensorflow as tf
import numpy as np
import random

class SRGAN():
    def __init__(self,generator_image_shape,discriminator_image_shape,scale_factor=4):
        self.__name__='SRGAN'
        self.generator_image_shape=generator_image_shape
        self.discriminator_image_shape=discriminator_image_shape
        self.scale_factor=scale_factor
    
    def Build_Generator(self):
        def Residual_Block(layer_input,filters,kernel_size):
            # x=tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(layer_input)
            x=tf.keras.layers.Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(layer_input)
            x=tf.keras.layers.PReLU()(x)
            x=tf.keras.layers.BatchNormalization(momentum=0.8)(x)
            # x=tf.keras.layers.Activation('relu')(x)
            # x=tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(x)
            x=tf.keras.layers.Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(x)
            x=tf.keras.layers.BatchNormalization(momentum=0.8)(x)
            x=tf.keras.layers.Add()([layer_input,x])
            return x
        
        def PixelShuffler(layer_input,filters,kernel_size):
            
            # x=tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(layer_input)
            x=tf.keras.layers.Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=1,padding='same')(layer_input)
            # *******Pixel Shuffler times 2********
            x=tf.keras.layers.Conv2D(filters=4*filters,kernel_size=kernel_size,strides=1,padding='same')(x)
            x=tf.nn.depth_to_space(x,2)
            # *************************************
            x=tf.keras.layers.PReLU()(x)
            # x=tf.keras.layers.Activation('relu')(x)
            return x
        
        input_image=tf.keras.layers.Input(shape=self.generator_image_shape,name='LR_Input')
        # x=tf.keras.layers.Conv2D(filters=64,kernel_size=9,strides=1,padding='same')(input_image)
        x=tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=9,strides=1,padding='same')(input_image)
        x=tf.keras.layers.PReLU()(x)
        # x=tf.keras.layers.Activation('relu')(x)
        for _ in range(5):
            x=Residual_Block(x,64,3)
        # x=tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same')(x)
        x=tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=1,padding='same')(x)
        x=tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x=tf.keras.layers.Add()([input_image,x])
        for _ in range(int(self.scale_factor/2)):
            x=PixelShuffler(x,256,3)
        x=tf.keras.layers.Conv2D(filters=1,kernel_size=9,strides=1,padding='same')(x)
        x=tf.keras.layers.Activation('tanh')(x)
        
        model=tf.keras.models.Model(inputs=input_image,outputs=x)
        model.summary()
        return model
    
    def Build_Discriminator(self):
        inputs=tf.keras.layers.Input(shape=self.discriminator_image_shape)
        kernel_size=5
        layer_filters=[32,64,128,256]
        x=inputs
        for filters in layer_filters:
            if filters==layer_filters[-1]:
                strides=1
            else:
                strides=2
            x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            x=tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same')(x)
        x=tf.keras.layers.Flatten()(x)
        x=tf.keras.layers.Dense(units=1)(x)
        x=tf.keras.layers.Activation(activation='sigmoid')(x)
        
        model=tf.keras.models.Model(inputs=inputs,outputs=x)
        model.summary()
        return model
    
    def Build_GAN(self):
        # Discriminator Model
        discriminator=self.Build_Discriminator()
        # optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002)
        lr=2e-4
        decay=6e-8
        optimizer=tf.keras.optimizers.RMSprop(lr=lr,decay=decay)
        discriminator.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        discriminator.summary()
        # Generator Model
        generator=self.Build_Generator()
        generator.summary()
        # Adversarial Model
        discriminator.trainable=False
        Ad_input=tf.keras.layers.Input(shape=self.generator_image_shape)
        adversarial=tf.keras.models.Model(inputs=Ad_input,outputs=discriminator(generator(Ad_input)))
        optimizer=tf.keras.optimizers.RMSprop(lr=0.5*lr,decay=0.5*decay)
        adversarial.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        adversarial.summary()
        return generator,discriminator,adversarial
        
    def Train_GAN(self,LR_Image_Patch,HR_Image_Patch,filepath,epochs,batch_size=64):
        generator,discriminator,adversarial=self.Build_GAN()
        train_size=LR_Image_Patch.shape[0]
        for epoch in range(epochs):
            for _ in range(5):
                rand_indexes=np.random.randint(0,train_size,size=batch_size)
                real_image=HR_Image_Patch[rand_indexes]
                LR_image=LR_Image_Patch[rand_indexes]
                fake_image=generator.predict(LR_image)
                x=np.concatenate((real_image,fake_image))
                y=np.ones((2*batch_size,1))
                y[batch_size:,:]=0.0
                discriminator.trainable=True
                loss,acc=discriminator.train_on_batch(x=x,y=y)
                d_log='%d: [discriminator loss: %f, acc: %f]'%(epoch,loss,acc)
            discriminator.trainable=False
            rand_indexes_adv=np.random.randint(0,train_size,size=batch_size)
            LR_image_adv=LR_Image_Patch[rand_indexes_adv]
            y=np.ones((batch_size,1))
            loss,acc=adversarial.train_on_batch(x=LR_image_adv,y=y)
            ad_log='%d: [adversarial loss: %f, acc: %f]'%(epoch,loss,acc)
            print(d_log,ad_log)

        generator.save(filepath=filepath+'.h5')
        generator.save_weights(filepath=filepath+'.hdf5')
        return generator,discriminator,adversarial
    
    
    
    
    
    
    
    
    
    
    
    
    
            