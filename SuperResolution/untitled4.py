#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 01:36:28 2021

@author: nephilim
"""
from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

dir_path='./Bicubic_FWI/20Hz_imodel_file'
file_num=int(len(list(Path(dir_path).iterdir()))/2)-1
data=np.load('./Bicubic_FWI/20Hz_imodel_file/%s_imodel.npy'%file_num)
data=data.reshape((240,440))
Bicubic=data[33:-20,20:-20]


dir_path='./4scale/20Hz_imodel_file'
file_num=int(len(list(Path(dir_path).iterdir()))/2)-1
data=np.load('./4scale/20Hz_imodel_file/%s_imodel.npy'%file_num)
data=data.reshape((240,440))
SSRGAN=data[33:-20,20:-20]

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