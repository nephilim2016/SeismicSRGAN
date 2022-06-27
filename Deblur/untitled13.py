#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 00:36:32 2021

@author: nephilim
"""

from pathlib import Path
import numpy as np
from skimage import filters
import skimage.transform
from matplotlib import pyplot,cm
from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def Error_(true,init,model):
    frac_up=np.linalg.norm(model-true,2)
    frac_down=np.linalg.norm(init-true,2)
    return frac_up/frac_down
if __name__ == '__main__':
    
    True_data=np.load('Marmousi.npy')*1333
    True_data=skimage.transform.resize(True_data,output_shape=(300,1000),mode='symmetric')
    
    Init_data=filters.gaussian(True_data,sigma=50)
    
    FWI_error=[]
    dir_path='./MarmousiFWI/5Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    FWI_5Hz_error=[]
    Init_data_5Hz=Init_data[::4,::4]
    True_data_5Hz=True_data[::4,::4]
    for idx in range(file_num):
        data=np.load('./MarmousiFWI/5Hz_imodel_file/%s_imodel.npy'%idx)
        data=data.reshape((115,-1))
        FWI_5Hz=data[20:-20,20:-20]
        error=Error_(True_data_5Hz,Init_data_5Hz,FWI_5Hz)
        FWI_5Hz_error.append(error)
        FWI_error.append(error)
        
    
    dir_path='./MarmousiFWI/10Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    FWI_10Hz_error=[]
    Init_data_10Hz=Init_data[::2,::2]
    True_data_10Hz=True_data[::2,::2]
    for idx in range(file_num):
        data=np.load('./MarmousiFWI/10Hz_imodel_file/%s_imodel.npy'%idx)
        data=data.reshape((190,-1))
        FWI_10Hz=data[20:-20,20:-20]
        error=Error_(True_data_10Hz,Init_data_10Hz,FWI_10Hz)
        FWI_10Hz_error.append(error)
        FWI_error.append(error)
    
    
    dir_path='./MarmousiFWI/20Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    FWI_20Hz_error=[]
    Init_data_20Hz=Init_data
    True_data_20Hz=True_data
    for idx in range(file_num):
        data=np.load('./MarmousiFWI/20Hz_imodel_file/%s_imodel.npy'%idx)
        data=data.reshape((340,-1))
        FWI_20Hz=data[20:-20,20:-20]
        error=Error_(True_data_20Hz,Init_data_20Hz,FWI_20Hz)
        FWI_20Hz_error.append(error)
        FWI_error.append(error)
    
    error_1=FWI_error+(1-np.max(FWI_error))
    import numpy as np

    
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
        
        
        
    fig=pyplot.figure()
    ax1=fig.add_subplot(111)
    lns1=pyplot.plot(data_FWI,'b--',label='Data Misfit')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Data Misfit')

    ax2=ax1.twinx() # this is the important function
    lns2=pyplot.plot(error_1,'r-',label='Reconstruction Error')
    pyplot.ylabel('Reconstruction Error')
    
    
    lns=lns1+lns2
    labs=[l.get_label() for l in lns]
    pyplot.legend(lns,labs,loc=0)
    pyplot.savefig('MarmousiMisfit.png',dpi=1000)
    
    
    