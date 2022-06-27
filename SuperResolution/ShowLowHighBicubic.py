#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 19:39:33 2021

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.transform


if __name__=='__main__':
    FileName='./OverThrustClip.npy'
    Marmousi=np.load(FileName)
    Scale4Marmousi=Marmousi[::4,::4]
    BicubicMarmousi=skimage.transform.resize(Scale4Marmousi,output_shape=Marmousi.shape,mode='symmetric')
    
    pyplot.figure(1)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1870,0]
    gci=pyplot.imshow(Marmousi,extent=extent,cmap=cm.seismic,norm=norm)
    # idx=np.arange(0,4000,10)
    # y=np.linspace(0,1870,50)
    # for idx_x in idx:
    #     if np.mod(idx_x,40)==0:
    #         pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
    #     else:
    #         pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
    # idx=np.arange(0,1870,10)
    # x=np.linspace(0,4000,50)
    # for idx_z in idx:
    #     if np.mod(idx_z,40)==0:
    #         pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
    #     else:
    #         pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
        
    # ax=pyplot.gca()
    # divider=make_axes_locatable(ax)
    # cax=divider.append_axes('right', size='3%', pad=0.35)
    # cbar=pyplot.colorbar(gci,cax=cax)
    # cbar.set_label('$m/s$')
    # ax.set_xlabel('Position (m)')
    # ax.set_ylabel('Depth (m)') 
    pyplot.axis('off')
    # pyplot.savefig('OverThrust_Result.png',dpi=1000)
    # pyplot.savefig('OverThrust_Result.svg',dpi=1000)
    pyplot.savefig('OverThrust_Result_Grid.png',dpi=1000)
    pyplot.savefig('OverThrust_Result_Grid.svg',dpi=1000)
    
    pyplot.figure(2)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1870,0]
    gci=pyplot.imshow(Scale4Marmousi,extent=extent,cmap=cm.seismic,norm=norm)
    idx=np.arange(0,4000,40)
    y=np.linspace(0,1870,50)
    for idx_x in idx:
        pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        
    idx=np.arange(0,1870,40)
    x=np.linspace(0,4000,50)
    for idx_z in idx:
        pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        
    # ax=pyplot.gca()
    # divider=make_axes_locatable(ax)
    # cax=divider.append_axes('right', size='3%', pad=0.35)
    # cbar=pyplot.colorbar(gci,cax=cax)
    # cbar.set_label('$m/s$')
    # ax.set_xlabel('Position (m)')
    # ax.set_ylabel('Depth (m)') 
    pyplot.axis('off')
    # pyplot.savefig('OverThrustScale4_Result_Grid.png',dpi=1000)
    # pyplot.savefig('OverThrustScale4_Result_Grid.svg',dpi=1000)
    # pyplot.savefig('MarmousiScale4_Result.png',dpi=1000)
    # pyplot.savefig('MarmousiScale4_Result.svg',dpi=1000)
    
    pyplot.figure(3)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1870,0]
    gci=pyplot.imshow(BicubicMarmousi,extent=extent,cmap=cm.seismic,norm=norm)
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
        
    # ax=pyplot.gca()
    # divider=make_axes_locatable(ax)
    # cax=divider.append_axes('right', size='3%', pad=0.35)
    # cbar=pyplot.colorbar(gci,cax=cax)
    # cbar.set_label('$m/s$')
    # ax.set_xlabel('Position (m)')
    # ax.set_ylabel('Depth (m)') 
    pyplot.axis('off')
    # pyplot.savefig('MarmousiBicubic_Result.png',dpi=1000)
    # pyplot.savefig('MarmousiBicubic_Result.svg',dpi=1000)
    