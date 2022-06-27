#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:04:45 2021

@author: nephilim
"""

import numpy as np

def DataNormalize(Data):
    MaxData=np.max(Data)
    MinData=np.min(Data)
    Data-=MinData
    Data/=(MaxData-MinData)
    return Data*255

def PSNR(X,Y):
    H,W=X.shape
    diff2=np.sum((X-Y)**2)
    MSE=diff2/(H*W)
    PSNR=10*np.log10((2**8-1)**2/MSE)
    return PSNR
    
def SSIM(X,Y):
    K1=0.01
    K2=0.03
    L=255
    C1=(K1*L)**2
    C2=(K2*L)**2
    C3=C2/2
    
    H,W=X.shape
    mu_x=np.sum(X)/(H*W)
    sigma_x=np.sqrt(np.sum((X-mu_x)**2)/(H*W-1))
    mu_y=np.sum(Y)/(H*W)
    sigma_y=np.sqrt(np.sum((Y-mu_y)**2)/(H*W-1))
    sigma_xy=np.sum((X-mu_x)*(Y-mu_y))/(H*W-1)
    
    l=(2*mu_x*mu_y+C1)/(mu_x**2+mu_y**2+C1)
    c=(2*sigma_x*sigma_y+C2)/(sigma_x**2+sigma_y**2+C2)
    s=(sigma_xy+C3)/(sigma_x*sigma_y+C3)
    SSIM=l*c*s
    return SSIM
    
    
    
if __name__=='__main__':
    # Clean_Image_=DataNormalize(HR_Image_)
    # Blur_Image_=DataNormalize(LR)
    # PredictData_=DataNormalize(PredictData)
    
    Clean_Image_=DataNormalize(Clean_Image)
    Blur_Image_=DataNormalize(Blur_Image)
    PredictData_=DataNormalize(PredictData)
    
    PSNR_1=PSNR(Clean_Image_,Blur_Image_)
    print('Blur PSNR: %s'%PSNR_1)
    
    PSNR_2=PSNR(Clean_Image_,PredictData_)
    print('Deblur PSNR: %s'%PSNR_2)
    
    SSIM_1=SSIM(Clean_Image_,Blur_Image_)
    print('Blur SSIM: %s'%SSIM_1)
    
    SSIM_2=SSIM(Clean_Image_,PredictData_)
    print('Deblur SSIM: %s'%SSIM_2)