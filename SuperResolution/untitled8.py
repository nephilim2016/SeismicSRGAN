#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 03:13:59 2021

@author: nephilim
"""


from pathlib import Path
import numpy as np
from skimage import filters
from matplotlib import pyplot,cm
from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def Error_(true,init,model):
    frac_up=np.linalg.norm(model-true,2)
    frac_down=np.linalg.norm(init-true,2)
    return frac_up/frac_down
if __name__ == '__main__':
    
    True_data=np.load('./Bicubic_FWI/20Hz_imodel_file/199_imodel.npy')
    True_data=True_data.reshape((240,440))
    True_data=True_data[33:-20,20:-20]
    Init_data=filters.gaussian(True_data,sigma=15)
    
    dir_path='./Bicubic_FWI/20Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    Bicubic_error=[]
    for idx in range(file_num):
        data=np.load('./Bicubic_FWI/20Hz_imodel_file/%s_imodel.npy'%idx)
        data=data.reshape((240,440))
        Bicubic=data[33:-20,20:-20]
        error=Error_(True_data,Init_data,Bicubic)
        Bicubic_error.append(error)
    
    True_data=np.load('./4scale/20Hz_imodel_file/199_imodel.npy')
    True_data=True_data.reshape((240,440))
    True_data=True_data[33:-20,20:-20]
    
    dir_path='./4scale/20Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    SSRGAN_error=[]
    for idx in range(file_num):

        data=np.load('./4scale/20Hz_imodel_file/%s_imodel.npy'%idx)
        data=data.reshape((240,440))
        SSRGAN=data[33:-20,20:-20]
        error=Error_(True_data,Init_data,SSRGAN)
        SSRGAN_error.append(error)
        
    error_1=SSRGAN_error+(1-np.max(SSRGAN_error))
    error_2=Bicubic_error+(1-np.max(Bicubic_error))
    
    pyplot.figure()
    pyplot.plot(error_1,'b--')
    pyplot.plot(error_2,'r-')
    pyplot.legend(['Bicubic Interpolation','SSRGAN'])
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Reconstruction Error')
    # pyplot.savefig('ReconstructionError.png',dpi=1000)
    
    
    
    


