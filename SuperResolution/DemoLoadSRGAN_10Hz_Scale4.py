#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:37:41 2021

@author: nephilim
"""

import numpy as np
import keras
import skimage.transform
import my_Im2col
import DataNormalized
from matplotlib import pyplot,cm,colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def Patch2TrainData(Patch,PatchSize):
    TrainData=np.zeros(((Patch.shape[1],)+PatchSize+(1,)))
    for idx in range(Patch.shape[1]):
        TrainData[idx,:,:,0]=Patch[:,idx].reshape(PatchSize)
    return TrainData

def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def PreProcess_Data(Data):
    PreData=np.zeros_like(Data)
    MaxData=[]
    MinData=[]
    for idx in range(Data.shape[0]):
        Data_tmp,MaxData_tmp,MinData_tmp=DataNormalized.DataNormalized(Data[idx,:,:,0])
        PreData[idx,:,:,0]=Data_tmp
        MaxData.append(MaxData_tmp)
        MinData.append(MinData_tmp)
    return MaxData,MinData,PreData

def PostProcess_Data(Data,MaxData,MinData):
    PostData=np.zeros_like(Data)
    for idx in range(Data.shape[0]):
        Data_tmp=DataNormalized.InverseDataNormalized(Data[idx,:,:],MaxData[idx],MinData[idx])
        PostData[idx,:,:]=Data_tmp
    return PostData

if __name__=='__main__':
    model=keras.models.load_model('ScaleFactor2TrainIteration60k.h5')

    ScaleFactor=2
    HR_patch_size=(64,64)
    HR_slidingDis=4
    LR_patch_size=(32,32)
    LR_slidingDis=2

    
    HR_Image_=np.load('./Marmousi.npy')*1333
    HR_Image=HR_Image_[::2,::2]
    LR_Image=np.load('./10Hz_Marmousi.npy')
    LR_Image=LR_Image.reshape((115,-1))
    LR_Image=LR_Image[20:-20,20:-20]
    
    Init_LR_Image=LR_Image
    Bucib_Image=skimage.transform.resize(LR_Image,output_shape=HR_Image_.shape,mode='symmetric')
    
    LR_Image_Patch_,Patch_Idx=GetPatch(LR_Image,LR_patch_size,LR_slidingDis)
    HR_Image_Patch_,Patch_Idx=GetPatch(HR_Image,HR_patch_size,HR_slidingDis)
    
    LR_Image_Patch=Patch2TrainData(LR_Image_Patch_,LR_patch_size)
    HR_Image_Patch=Patch2TrainData(HR_Image_Patch_,HR_patch_size)
    
    MaxDataInput,MinDataInput,LR_Image_Patch=PreProcess_Data(LR_Image_Patch)
    MaxDataOutput,MinDataOutput,HR_Image_Patch=PreProcess_Data(HR_Image_Patch)
    
    Predict_Patch=model.predict(LR_Image_Patch)
    x_decoded=Predict_Patch[:,:,:,0]
    x_decoded=PostProcess_Data(x_decoded,MaxDataOutput,MinDataOutput)
    
    rows,cols=my_Im2col.ind2sub(np.array(HR_Image.shape)-HR_patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(HR_Image.shape)
    Weight=np.zeros(HR_Image.shape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=block
        Weight[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=np.ones(HR_patch_size)
        count+=1
    PredictData=IMout/Weight

    LR_Image=PredictData
    HR_Image=skimage.transform.resize(LR_Image,output_shape=(300,1000),mode='symmetric')
    LR_Image_Patch_,Patch_Idx=GetPatch(LR_Image,LR_patch_size,LR_slidingDis)
    HR_Image_Patch_,Patch_Idx=GetPatch(HR_Image,HR_patch_size,HR_slidingDis)
    
    LR_Image_Patch=Patch2TrainData(LR_Image_Patch_,LR_patch_size)
    HR_Image_Patch=Patch2TrainData(HR_Image_Patch_,HR_patch_size)
    
    MaxDataInput,MinDataInput,LR_Image_Patch=PreProcess_Data(LR_Image_Patch)
    MaxDataOutput,MinDataOutput,HR_Image_Patch=PreProcess_Data(HR_Image_Patch)
    
    Predict_Patch=model.predict(LR_Image_Patch)
    x_decoded=Predict_Patch[:,:,:,0]
    x_decoded=PostProcess_Data(x_decoded,MaxDataOutput,MinDataOutput)
    
    rows,cols=my_Im2col.ind2sub(np.array(HR_Image.shape)-HR_patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(HR_Image.shape)
    Weight=np.zeros(HR_Image.shape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=block
        Weight[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=np.ones(HR_patch_size)
        count+=1
    PredictData=IMout/Weight


    LR=skimage.transform.resize(Init_LR_Image,output_shape=HR_Image.shape,mode='symmetric')
    pyplot.figure(1)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1880,0]
    gci=pyplot.imshow(Init_LR_Image,extent=extent,cmap=cm.seismic,norm=norm)
    idx=np.arange(0,4000,40)
    y=np.linspace(0,1880,50)
    for idx_x in idx:
        pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        
    idx=np.arange(0,1880,40)
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
    pyplot.savefig('10Hz_Scale4_Result.png',dpi=1000)
    pyplot.savefig('10Hz_Scale4_Result.svg',dpi=1000)
    
    
    pyplot.figure(2)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1870,0]
    gci=pyplot.imshow(LR,extent=extent,cmap=cm.seismic,norm=norm)
    
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
    pyplot.savefig('FWI_Bicubic_10Hz.png',dpi=1000)
    pyplot.savefig('10Hz_Scale1_Bicubic_Result.svg',dpi=1000)
    
    
    pyplot.figure(3)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1870,0]
    gci=pyplot.imshow(PredictData,extent=extent,cmap=cm.seismic,norm=norm)
    
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
    pyplot.savefig('FWI_SRGAN_10Hz.png',dpi=1000)
    pyplot.savefig('10Hz_Scale1_SRGAN_Result.svg',dpi=1000)
    
    pyplot.figure(4)
    index=190
    vp_true_line=HR_Image[:,index]
    vp_10Hz_line=Init_LR_Image[:,int(index/4)]
    vp_10Hz_line_=np.zeros(188)
    for idx,data in enumerate(vp_10Hz_line):
        vp_10Hz_line_[4*idx]=data
        vp_10Hz_line_[4*idx+1]=data
        vp_10Hz_line_[4*idx+2]=data
        vp_10Hz_line_[4*idx+3]=data
    vp_10Hz_line=np.zeros(187)
    vp_10Hz_line=vp_10Hz_line_[:-1]

    Bicubic_line=Bucib_Image[:,index]
    SRGAN_line=PredictData[:,index]

    pyplot.plot(vp_true_line,np.linspace(0,1870,187),'k--')
    pyplot.plot(vp_10Hz_line,np.linspace(0,1870,187),'g:')
    pyplot.plot(Bicubic_line,np.linspace(0,1870,187),'b-.')
    pyplot.plot(SRGAN_line,np.linspace(0,1870,187),'r-')

    ax=pyplot.gca()
    ax.invert_yaxis()
    ax.set(aspect=3)
    
    pyplot.legend(['True Model','Low Resolution','Bicubic Interpolation','SRGAN'],frameon=False)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Depth (m)') 
    pyplot.savefig('10Hz_Line_Result.png',dpi=1000)
    pyplot.savefig('10Hz_Line_Result.svg',dpi=1000)
    
    # pyplot.figure(4)
    # # pyplot.plot(LR_Image[:,100])
    # pyplot.plot(HR_Image[:,200],'b--')
    # pyplot.plot(LR[:,200],'k:')
    # pyplot.plot(PredictData[:,200],'r-')
    
