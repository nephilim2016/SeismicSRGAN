#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 23:52:43 2021

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
    dir_path='./MarmousiFWI/5Hz_imodel_file'
    d0=np.load('./MarmousiFWI/5Hz_imodel_file/0_info.npy')
    d0_FWI=d0[3]
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    data_num=[]
    data_FWI=[]
    for iter_ in range(file_num):
        data=np.load('./MarmousiFWI/5Hz_imodel_file/%s_info.npy'%iter_)
        data_num.append(data[1])
        data_FWI.append(data[3]/d0_FWI)
    
    dir_path='./MarmousiFWI/10Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    for iter_ in range(file_num):
        data=np.load('./MarmousiFWI/10Hz_imodel_file/%s_info.npy'%iter_)
        data_num.append(data[1])
        data_FWI.append(data[3]/d0_FWI)
    
    ###########################################################################
    
    dir_path='./MarmousiFWI/20Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    for iter_ in range(file_num):
        data=np.load('./MarmousiFWI/20Hz_imodel_file/%s_info.npy'%iter_)
        data_num.append(data[1])
        data_FWI.append(data[3]/d0_FWI)
        

    
    
    pyplot.figure()
    pyplot.plot(data_FWI,'r-')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Data Misfit')
    pyplot.savefig('Marmousi_Data_Misfit.png',dpi=1000)
    