#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 02:10:02 2021

@author: nephilim
"""

sigma=5


# pyplot.figure(1)
# index=50
# norm=colors.Normalize(vmin=2500, vmax=6000)
# extent=[index*10,index*10+320,index*10+320,index*10]
# gci=pyplot.imshow(Clean_Image[index:index+32,index:index+32],extent=extent,cmap=cm.seismic,norm=norm)
# ax=pyplot.gca()
# divider=make_axes_locatable(ax)
# cax=divider.append_axes('right', size='7%', pad=0.15)
# cbar=pyplot.colorbar(gci,cax=cax)
# cbar.set_label('$m/s$')
# ax.set_xlabel('Position (m)')
# ax.set_ylabel('Depth (m)')
# pyplot.savefig('Clean_Patch_Result.png',dpi=1000)

pyplot.figure(2)
index=50
norm=colors.Normalize(vmin=2500, vmax=6000)
extent=[index*10,index*10+320,index*10+320,index*10]
gci=pyplot.imshow(Blur_Image[index:index+32,index:index+32],extent=extent,cmap=cm.seismic,norm=norm)
ax=pyplot.gca()
divider=make_axes_locatable(ax)
cax=divider.append_axes('right', size='7%', pad=0.15)
cbar=pyplot.colorbar(gci,cax=cax)
cbar.set_label('$m/s$')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Depth (m)')
pyplot.savefig('sigma_%s_Blur_Patch_Result.png'%sigma,dpi=1000)

pyplot.figure(3)
index=50
norm=colors.Normalize(vmin=2500, vmax=6000)
extent=[index*10,index*10+320,index*10+320,index*10]
gci=pyplot.imshow(PredictData[index:index+32,index:index+32],extent=extent,cmap=cm.seismic,norm=norm)
ax=pyplot.gca()
divider=make_axes_locatable(ax)
cax=divider.append_axes('right', size='7%', pad=0.15)
cbar=pyplot.colorbar(gci,cax=cax)
cbar.set_label('$m/s$')
ax.set_xlabel('Position (m)')
ax.set_ylabel('Depth (m)')
pyplot.savefig('sigma_%s_Deblur_Patch_Result.png'%sigma,dpi=1000)