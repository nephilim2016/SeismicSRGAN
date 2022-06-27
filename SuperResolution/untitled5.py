#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 01:58:36 2021

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
    
    
    data=np.load('./Bicubic_FWI/20Hz_imodel_file/199_imodel.npy')
    data=data.reshape((240,440))
    Bicubic=data[33:-20,20:-20]
    pyplot.figure(1)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1870,0]
    gci=pyplot.imshow(Bicubic,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,4000,10)
    y=np.linspace(0,1870,50)
    for idx_x in idx:
        if np.mod(idx_x,40)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        else:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
    idx=np.arange(0,1870,10)
    x=np.linspace(0,4000,50)
    for idx_z in idx:
        if np.mod(idx_z,40)==0:
            pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        else:
            pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
        
    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='3%', pad=0.35)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    # pyplot.savefig('Bicubic_20Hz_FWI.png',dpi=1000)

    
    data=np.load('./4scale/20Hz_imodel_file/199_imodel.npy')
    data=data.reshape((240,440))
    SSRGAN=data[33:-20,20:-20]
    pyplot.figure(2)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1870,0]
    gci=pyplot.imshow(SSRGAN,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,4000,10)
    y=np.linspace(0,1870,50)
    for idx_x in idx:
        if np.mod(idx_x,40)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        else:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
    idx=np.arange(0,1870,10)
    x=np.linspace(0,4000,50)
    for idx_z in idx:
        if np.mod(idx_z,40)==0:
            pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        else:
            pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
        
    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='3%', pad=0.35)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    # pyplot.savefig('SSRGAN_20Hz_FWI.png',dpi=1000)
    
    
    FWI_data_true=np.load('OverThrustClip.npy')
    
    FWI_data_Bicubic_20Hz_FWI=np.load('./Bicubic_FWI/20Hz_imodel_file/99_imodel.npy')
    data=FWI_data_Bicubic_20Hz_FWI.reshape((240,440))
    FWI_data_Bicubic_20Hz_FWI=data[33:-20,20:-20]
    
    FWI_data_SSRGAN_20Hz_FWI=np.load('./4scale/20Hz_imodel_file/199_imodel.npy')
    data=FWI_data_SSRGAN_20Hz_FWI.reshape((240,440))
    FWI_data_SSRGAN_20Hz_FWI=data[33:-20,20:-20]
    
    pyplot.figure()
    index=190
    vp_true_line=FWI_data_true[:,index]
    vp_Bicubic_20Hz_FWI_line=FWI_data_Bicubic_20Hz_FWI[:,index]
    vp_SSRGAN_20Hz_FWI_line=FWI_data_SSRGAN_20Hz_FWI[:,index]

    vp_true_Bicubic_line=FWI_data_Bicubic[:,index]
    SRGAN_line=PredictData[:,index]
    
    
    # pyplot.plot(vp_true_line,np.linspace(0,1870,187),'k--')
    # pyplot.plot(vp_true_Bicubic_line[::2],np.linspace(0,1870,187),'g-.')
    # pyplot.plot(SRGAN_line[::2],np.linspace(0,1870,187),'b-.')
    # pyplot.plot(vp_Bicubic_20Hz_FWI_line,np.linspace(0,1870,187),'c-')
    # pyplot.plot(vp_SSRGAN_20Hz_FWI_line,np.linspace(0,1870,187),'r-')
    
    pyplot.plot(np.linspace(0,1870,187),vp_true_line,'k--')
    pyplot.plot(np.linspace(0,1870,187),vp_true_Bicubic_line[::2],'g-.')
    pyplot.plot(np.linspace(0,1870,187),SRGAN_line[::2],'b-.')
    pyplot.plot(np.linspace(0,1870,187),vp_Bicubic_20Hz_FWI_line,'c-')
    pyplot.plot(np.linspace(0,1870,187),vp_SSRGAN_20Hz_FWI_line,'r-')
    
    ax=pyplot.gca()
    # ax.invert_yaxis()
    ax.set(aspect=0.2)
    pyplot.legend(['True Model','Bicubic Interpolation (10Hz)','SSRGAN (10Hz)','Bicubic Interpolation (20Hz)','SSRGAN (20Hz)'],frameon=False)
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Depth (m)') 
    pyplot.savefig('New_FWI_20Hz_Line.png',dpi=1000)

