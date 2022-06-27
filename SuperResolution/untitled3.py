#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:34:11 2021

@author: nephilim
"""
from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

############True############
pyplot.figure(1)
index=50
norm=colors.Normalize(vmin=2500, vmax=6000)
extent=[index*10,index*10+320,index*10+320,index*10]
gci=pyplot.imshow(FWI_data_true[index:index+32,index:index+32],extent=extent,cmap=cm.seismic,norm=norm)

idx=np.arange(500,820,10)
y=np.linspace(500,820,50)
for idx_x in idx:
    if np.mod(idx_x,40)==20:
        pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
    else:
        pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
idx=np.arange(500,820,10)
x=np.linspace(500,820,50)
for idx_z in idx:
    if np.mod(idx_z,40)==20:
        pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
    else:
        pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
    
ax=pyplot.gca()
divider=make_axes_locatable(ax)
cax=divider.append_axes('right', size='7%', pad=0.15)
cbar=pyplot.colorbar(gci,cax=cax)
cbar.set_label('$m/s$')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Depth (m)')
pyplot.savefig('True_Patch.png',dpi=1000)

############LR############
pyplot.figure(2)
index=14
norm=colors.Normalize(vmin=2500, vmax=6000)
extent=[500,820,820,500]
gci=pyplot.imshow(FWI_data_LR[index:index+8,index:index+8],extent=extent,cmap=cm.seismic,norm=norm)

idx=np.arange(500,820,10)
y=np.linspace(500,820,50)
for idx_x in idx:
    if np.mod(idx_x,40)==20:
        pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        
idx=np.arange(500,820,10)
x=np.linspace(500,820,50)
for idx_z in idx:
    if np.mod(idx_z,40)==20:
        pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        
ax=pyplot.gca()
divider=make_axes_locatable(ax)
cax=divider.append_axes('right', size='7%', pad=0.15)
cbar=pyplot.colorbar(gci,cax=cax)
cbar.set_label('$m/s$')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Depth (m)')
pyplot.savefig('10Hz_LR_Patch.png',dpi=1000)

############HR############
pyplot.figure(3)
index=50
norm=colors.Normalize(vmin=2500, vmax=6000)
extent=[index*10,index*10+320,index*10+320,index*10]
gci=pyplot.imshow(FWI_data_FWI[index:index+32,index:index+32],extent=extent,cmap=cm.seismic,norm=norm)

idx=np.arange(500,820,10)
y=np.linspace(500,820,50)
for idx_x in idx:
    if np.mod(idx_x,40)==20:
        pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
    else:
        pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
idx=np.arange(500,820,10)
x=np.linspace(500,820,50)
for idx_z in idx:
    if np.mod(idx_z,40)==20:
        pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
    else:
        pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
        
ax=pyplot.gca()
divider=make_axes_locatable(ax)
cax=divider.append_axes('right', size='7%', pad=0.15)
cbar=pyplot.colorbar(gci,cax=cax)
cbar.set_label('$m/s$')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Depth (m)')
pyplot.savefig('10Hz_HR_Patch.png',dpi=1000)

############Bicubic############
pyplot.figure(4)
index=50
norm=colors.Normalize(vmin=2500, vmax=6000)
extent=[index*10,index*10+320,index*10+320,index*10]
gci=pyplot.imshow(FWI_data_Bicubic[index:index+32,index:index+32],extent=extent,cmap=cm.seismic,norm=norm)

idx=np.arange(500,820,10)
y=np.linspace(500,820,50)
for idx_x in idx:
    if np.mod(idx_x,40)==20:
        pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
    else:
        pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
idx=np.arange(500,820,10)
x=np.linspace(500,820,50)
for idx_z in idx:
    if np.mod(idx_z,40)==20:
        pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
    else:
        pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
        
ax=pyplot.gca()
divider=make_axes_locatable(ax)
cax=divider.append_axes('right', size='7%', pad=0.15)
cbar=pyplot.colorbar(gci,cax=cax)
cbar.set_label('$m/s$')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Depth (m)')
pyplot.savefig('10Hz_Bicubic_Patch.png',dpi=1000)

############SSRGAN############
pyplot.figure(5)
index=50
norm=colors.Normalize(vmin=2500, vmax=6000)
extent=[index*10,index*10+320,index*10+320,index*10]
gci=pyplot.imshow(PredictData[index:index+32,index:index+32],extent=extent,cmap=cm.seismic,norm=norm)

idx=np.arange(500,820,10)
y=np.linspace(500,820,50)
for idx_x in idx:
    if np.mod(idx_x,40)==20:
        pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
    else:
        pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
idx=np.arange(500,820,10)
x=np.linspace(500,820,50)
for idx_z in idx:
    if np.mod(idx_z,40)==20:
        pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
    else:
        pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
        
ax=pyplot.gca()
divider=make_axes_locatable(ax)
cax=divider.append_axes('right', size='7%', pad=0.15)
cbar=pyplot.colorbar(gci,cax=cax)
cbar.set_label('$m/s$')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Depth (m)')
pyplot.savefig('10Hz_SSRGAN_Patch.png',dpi=1000)
