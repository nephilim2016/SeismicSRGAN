#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 22:51:42 2021

@author: nephilim
"""

index=1200
pyplot.figure()
pyplot.imshow(Blur_Image_Patch[index,:,:,0],cmap=cm.seismic)
pyplot.axis('off')
pyplot.savefig('%s_Blur.png'%index,dpi=1000)

pyplot.figure()
pyplot.imshow(Clean_Image_Patch[index,:,:,0],cmap=cm.seismic)
pyplot.axis('off')
pyplot.savefig('%s_Clean.png'%index,dpi=1000)
