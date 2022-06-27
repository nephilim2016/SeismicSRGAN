#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:06:16 2021

@author: nephilim
"""

import tensorflow as tf
import numpy as np
from numba import jit


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
        
        self.flatten=tf.keras.layers.Flatten()
        
        self.dense0=tf.keras.layers.Dense(units=1024)
        
        self.dense1=tf.keras.layers.Dense(units=1)
    
    def call(self,inputs,training=None):
        x=tf.nn.leaky_relu(self.Conv0(inputs))
        
        x=tf.nn.leaky_relu(self.bn1(self.Conv1(x),training=training))
        x=tf.nn.leaky_relu(self.bn2(self.Conv2(x),training=training))
        x=tf.nn.leaky_relu(self.bn3(self.Conv3(x),training=training))
        x=tf.nn.leaky_relu(self.bn4(self.Conv4(x),training=training))
        x=tf.nn.leaky_relu(self.bn5(self.Conv5(x),training=training))
        x=self.flatten(x)
        x=tf.nn.leaky_relu(self.dense0(x))
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

def gradient_penalty(discriminator,Real_image,Fake_image):

    batchsz=Real_image.shape[0]
    t=tf.random.uniform([batchsz,1,1,1])
    t=tf.broadcast_to(t,Real_image.shape)

    interplate=t*Real_image+(1-t)*Fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits=discriminator(interplate,training=True)
    grads=tape.gradient(d_interplote_logits,interplate)

    grads=tf.reshape(grads,[grads.shape[0],-1])
    gp=tf.norm(grads,axis=1)
    gp=tf.reduce_mean((gp-1)**2)
    return gp

def d_loss_fn(generator,discriminator,LR_image,Real_image,is_training):
    fake_image=generator(LR_image,is_training)
    d_fake_logits=discriminator(fake_image,is_training)
    d_real_logits=discriminator(Real_image,is_training)
    gp=gradient_penalty(discriminator,Real_image,fake_image)
    loss=tf.reduce_mean(d_fake_logits)-tf.reduce_mean(d_real_logits)+5.0*gp
    return loss,gp

def g_loss_fn(generator,discriminator,LR_image,is_training):
    fake_image=generator(LR_image,is_training)
    d_fake_logits=discriminator(fake_image,is_training)
    loss=-tf.reduce_mean(d_fake_logits)
    return loss  


@jit(nopython=True)
def Create_LR_Image(real_image):
    LR_image=np.zeros((60000,14,14,1))
    for idx in range(60000):
        LR_image[idx,:,:,0]=real_image[idx,::2,::2,0]
    return LR_image



if __name__=='__main__':
    (Real_Image,_),(_,_)=tf.keras.datasets.mnist.load_data()
    Real_Image=Real_Image/127.5-1
    Real_Image=Real_Image.reshape(-1,28,28,1)
    LR_Image=Create_LR_Image(Real_Image)
    
    batch_size=64
    generator=Generator()
    generator.build(input_shape=(batch_size,14,14,1))
    generator.summary()
    discriminator=Discriminator()
    discriminator.build(input_shape=(batch_size,28,28,1))
    discriminator.summary()
    
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003,beta_1=0.5)
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003,beta_1=0.5)
    
    train_size=LR_Image.shape[0]
    epoches=10
    d_losses=[]
    g_losses=[]
    for epoch in range(epoches):
        for _ in range(3):
            rand_indexes=np.random.randint(0,train_size,size=batch_size)
            LR_image=LR_Image[rand_indexes]
            Real_image=Real_Image[rand_indexes]
            with tf.GradientTape() as tape:
                d_loss,gp=d_loss_fn(generator,discriminator,LR_image,Real_image,is_training=True)
            grads=tape.gradient(d_loss,discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads,discriminator.trainable_variables))
        
        rand_indexes=np.random.randint(0,train_size,size=batch_size)
        LR_image=LR_Image[rand_indexes]
        Real_image=Real_Image[rand_indexes]
        with tf.GradientTape() as tape:
            g_loss=g_loss_fn(generator,discriminator,LR_image,is_training=True)
        grads=tape.gradient(g_loss,generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads,generator.trainable_variables))
        
        if epoch%1==0:
            print(epoch, 'd-loss:',float(d_loss), 'g-loss:', float(g_loss))

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

            if epoch%100==1:
                generator.save_weights('generator.ckpt')
                discriminator.save_weights('discriminator.ckpt')
    
    

    test=LR_Image[0,:,:,0].reshape((1,14,14,1))
    fake_image=generator(test)
    fake_image=fake_image.numpy()
    test=LR_Image[0,:,:,0].reshape((1,14,14,1))
    fake_image=generator(test)
    fake_image=fake_image.numpy()
    fake_image=fake_image[0,:,:,0]
    lr_image=test[0,:,:,0]
    real_image=Real_Image[0,:,:,0]
