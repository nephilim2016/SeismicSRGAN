#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:45:29 2020

@author: nephilim
"""
import numpy as np
import my_Im2col
from matplotlib import pyplot,cm
import skimage.transform


def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def GetPatchData(ProfileTarget,patch_size,slidingDis):
    Patch,Patch_Idx=GetPatch(ProfileTarget,patch_size,slidingDis)
    data=np.zeros((Patch.shape[1],patch_size[0],patch_size[1]))
    for idx in range(Patch.shape[1]):
        data[idx,:,:]=Patch[:,idx].reshape(patch_size)
    return data

def DisplayPatch(Patch,numRows,numCols,SizeForEachImage):
    bb=2
    I=np.ones((SizeForEachImage*numRows+bb,SizeForEachImage*numCols+bb))*(-1e6)
    maxRandom=Patch.shape[0]
    index=np.random.randint(0,maxRandom,size=maxRandom)
    counter=0
    for j in range(numRows):
        for i in range(numCols):
            I[bb+j*SizeForEachImage:(j+1)*SizeForEachImage+bb-2,bb+i*SizeForEachImage:(i+1)*SizeForEachImage+bb-2]=Patch[index[counter]]
            counter+=1
    # I-=np.min(I)
    # I/=np.max(I)
    # INoise-=np.min(INoise)
    # INoise/=np.max(INoise)
    return I
    
    

if __name__=='__main__':
    FileName='./SeismicResize/BP94.npy'
    Marmousi=np.load(FileName)
    Scale4Marmousi=Marmousi[::4,::4]
    BicubicMarmousi=skimage.transform.resize(Scale4Marmousi,output_shape=Marmousi.shape,mode='symmetric')
    
    patch_size=(64,64)
    slidingDis=32
    SizeForEachImage=66
    
    HR_Patch=GetPatchData(Marmousi,patch_size,slidingDis)
    
    # I=DisplayPatch(HR_Patch,16,16,SizeForEachImage)
    # pyplot.figure()
    # # pyplot.subplot(1,2,1)
    # pyplot.imshow(I,vmin=1200,vmax=6000,cmap=cm.seismic)
    # pyplot.axis('off')
    # # pyplot.savefig('PatchMarmousi.png',dpi=1000)
    # # pyplot.savefig('PatchMarmousi.png',dpi=1000)
    
    patch_size=(16,16)
    slidingDis=8
    SizeForEachImage=34
    
    LR_Patch=GetPatchData(Scale4Marmousi,patch_size,slidingDis)
    
    # I=DisplayPatch(LR_Patch,16,16,SizeForEachImage)
    # pyplot.figure()
    # # pyplot.subplot(1,2,1)
    # pyplot.imshow(I,vmin=1200,vmax=6000,cmap=cm.seismic)
    # pyplot.axis('off')
    # # pyplot.savefig('PatchScale4Marmousi.png',dpi=1000)
    # # pyplot.savefig('PatchScale4Marmousi.png',dpi=1000)
    
    
    
    index=92
    pyplot.figure()
    pyplot.imshow(LR_Patch[index],cmap=cm.seismic,extent=(0,16,0,16))
    idx=np.arange(0,16)
    y=np.linspace(0,16,50)
    for idx_x in idx:
        if np.mod(idx_x,1)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        else:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
    idx=np.arange(0,16)
    x=np.linspace(0,16,50)
    for idx_z in idx:
        if np.mod(idx_z,1)==0:
            pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        else:
            pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
    pyplot.axis('off')
    pyplot.savefig('%s_LR.png'%index,dpi=1000)
    
    pyplot.figure()
    pyplot.imshow(HR_Patch[index],cmap=cm.seismic,extent=(0,64,0,64))
    idx=np.arange(0,64)
    y=np.linspace(0,64,50)
    for idx_x in idx:
        if np.mod(idx_x,4)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        else:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
    idx=np.arange(0,64)
    x=np.linspace(0,64,50)
    for idx_z in idx:
        if np.mod(idx_z,4)==0:
            pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)
        else:
            pyplot.plot(x,[idx_z]*50,'k--',linewidth=0.1)
    pyplot.axis('off')
    pyplot.savefig('%s_HR.png'%index,dpi=1000)
    
    
    
    
    
    
    
    