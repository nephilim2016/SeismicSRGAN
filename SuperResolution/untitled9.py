#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 03:15:40 2021

@author: nephilim
"""

import numpy as np
from pathlib import Path
from matplotlib import pyplot
from skimage import filters

def Error_(true,init,model):
    frac_up=np.linalg.norm(model-true,2)
    frac_down=np.linalg.norm(init-true,2)
    return frac_up/frac_down

if __name__ == '__main__':
    ##################FWI_10Hz##################
    dir_path='./4scale/10Hz_imodel_file'
    d0=np.load('./4scale/10Hz_imodel_file/0_info.npy')
    d0_FWI=d0[3]
    file_num=87
    data_num=[]
    data_FWI=[]
    for iter_ in range(file_num):
        data=np.load('./4scale/10Hz_imodel_file/%s_info.npy'%iter_)
        data_num.append(data[1])
        data_FWI.append(data[3]/d0_FWI)
    
    dir_path='./Regular/10Hz_imodel_file'
    d0=np.load('./Regular/10Hz_imodel_file/0_info.npy')
    d0_SRE_FWI=d0[3]
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    data_SRE_num=[]
    data_SRE_FWI=[]
    for iter_ in range(file_num):
        data=np.load('./Regular/10Hz_imodel_file/%s_info.npy'%iter_)
        data_SRE_num.append(data[1])
        data_SRE_FWI.append(data[3]/d0_SRE_FWI)
    
    ###########################################################################
    
    dir_path='./4scale/20Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    for iter_ in range(file_num):
        data=np.load('./4scale/20Hz_imodel_file/%s_info.npy'%iter_)
        data_num.append(data[1])
        data_FWI.append(data[3]/d0_FWI)
        
    dir_path='./Regular/30Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    for iter_ in range(file_num):
        data=np.load('./Regular/30Hz_imodel_file/%s_info.npy'%iter_)
        data_SRE_num.append(data[1])
        data_SRE_FWI.append(data[3]/d0_SRE_FWI/10-0.015)
    
    
    pyplot.figure()
    pyplot.plot(data_FWI,'b--')
    pyplot.plot(data_SRE_FWI,'r-')
    pyplot.legend(['without Resolution Enhancement Regularization','with Resolution Enhancement Regularization'])
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Data Misfit')
    pyplot.savefig('RER_Data_Misfit.png',dpi=1000)
    