#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:44:10 2021

@author: nephilim
"""

import numpy as np
from pathlib import Path
from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__=='__main__':
    frequence=10
    dir_path='./%sHz_imodel_file'%frequence
    file_num=int(len(list(Path(dir_path).iterdir()))/2)-1
    data_10Hz=np.load('./%sHz_imodel_file/%s_imodel.npy'%(frequence,file_num))
    vp_10Hz=data_10Hz.reshape((140,-1))
    vp_10Hz=vp_10Hz[26:-20,20:-20]
    norm=colors.Normalize(vmin=2500,vmax=6000)
    extent=[0,4000,1880,0]
    
    pyplot.figure(1)
    gci=pyplot.imshow(vp_10Hz,extent=extent,cmap=cm.seismic, norm=norm)
    
    idx=np.arange(0,4000,20)
    y=np.linspace(0,1880,50)
    for idx_x in idx:
        pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        
    idx=np.arange(0,1880,20)
    x=np.linspace(0,4000,50)
    for idx_z in idx:
        pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        
    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='3%', pad=0.35)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    pyplot.savefig('10Hz_Result.png',dpi=1000)
    pyplot.savefig('10Hz_Result.svg',dpi=1000)
    
    frequence=20
    dir_path='./%sHz_imodel_file'%frequence
    file_num=int(len(list(Path(dir_path).iterdir()))/2)-1
    data_20Hz=np.load('./%sHz_imodel_file/%s_imodel.npy'%(frequence,file_num))
    vp_20Hz=data_20Hz.reshape((240,-1))
    vp_20Hz=vp_20Hz[33:-20,20:-20]
    norm=colors.Normalize(vmin=2500,vmax=6000)
    extent=[0,4000,1870,0]
    
    pyplot.figure(2)
    gci=pyplot.imshow(vp_20Hz,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,4000,10)
    y=np.linspace(0,1870,50)
    for idx_x in idx:
        if np.mod(idx_x,20)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        else:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
    idx=np.arange(0,1870,10)
    x=np.linspace(0,4000,50)
    for idx_z in idx:
        if np.mod(idx_z,20)==0:
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
    pyplot.savefig('20Hz_Result.png',dpi=1000)
    pyplot.savefig('20Hz_Result.svg',dpi=1000)
    
    pyplot.figure(3)
    index=190
    vp_true=np.load('OverThrustClip.npy')
    vp_true_line=vp_true[:,index]
    vp_10Hz_line=vp_10Hz[:,int(index/2)]
    vp_10Hz_line_=np.zeros(188)
    for idx,data in enumerate(vp_10Hz_line):
        vp_10Hz_line_[2*idx]=data
        vp_10Hz_line_[2*idx+1]=data
    vp_10Hz_line=np.zeros(187)
    vp_10Hz_line=vp_10Hz_line_[:-1]
        
    vp_20Hz_line=vp_20Hz[:,index]
    
    pyplot.plot(vp_true_line,np.linspace(0,1870,187),'k--')
    pyplot.plot(vp_10Hz_line,np.linspace(0,1870,187),'b-.')
    pyplot.plot(vp_20Hz_line,np.linspace(0,1870,187),'r-')
    
    ax=pyplot.gca()
    ax.invert_yaxis()
    ax.set(aspect=3)
    
    pyplot.legend(['True Model','FWI (10Hz)','FWI (20Hz)'],frameon=False)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Depth (m)') 
    pyplot.savefig('Line_Result.png',dpi=1000)
    pyplot.savefig('Line_Result.svg',dpi=1000)