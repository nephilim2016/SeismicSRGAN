#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:32:24 2022

@author: nephilim
"""

from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import skimage.transform
############True############
True_data=np.load('Marmousi.npy')*1333
True_data=skimage.transform.resize(True_data,output_shape=(300,1000),mode='symmetric')
    
pyplot.figure(1)
norm=colors.Normalize(vmin=1500, vmax=6000)
extent=[0,10000,3000,0]
gci=pyplot.imshow(True_data,extent=extent,cmap=cm.seismic,norm=norm)

idx=np.arange(0,10000,10)
y=np.linspace(0,3000,50)
# for idx_x in idx:
#     if np.mod(idx_x,4)==0:
#         pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
#     else:
#         pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
idx=np.arange(0,3000,10)
x=np.linspace(0,10000,50)
# for idx_z in idx:
#     if np.mod(idx_z,4)==0:
#         pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
#     else:
#         pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
    
ax=pyplot.gca()
divider=make_axes_locatable(ax)
cax=divider.append_axes('right', size='3%', pad=0.15)
cbar=pyplot.colorbar(gci,cax=cax)
cbar.set_label('$m/s$')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Depth (m)')
pyplot.savefig('True_Marmousi.png',dpi=1000)