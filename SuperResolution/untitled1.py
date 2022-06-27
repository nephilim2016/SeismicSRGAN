#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:02:13 2021

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

    
    HR_Image=np.load('./OverThrustClip.npy')
    HR_Image=skimage.transform.resize(HR_Image,output_shape=(374,800),mode='symmetric')
    LR_Image=np.load('./10Hz_FWI.npy')
    LR_Image=LR_Image.reshape((240,-1))
    LR_Image=LR_Image[33:-20,20:-20]
    
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

    # LR_Image=PredictData
    # HR_Image=skimage.transform.resize(LR_Image,output_shape=(187,400),mode='symmetric')
    # LR_Image_Patch_,Patch_Idx=GetPatch(LR_Image,LR_patch_size,LR_slidingDis)
    # HR_Image_Patch_,Patch_Idx=GetPatch(HR_Image,HR_patch_size,HR_slidingDis)
    
    # LR_Image_Patch=Patch2TrainData(LR_Image_Patch_,LR_patch_size)
    # HR_Image_Patch=Patch2TrainData(HR_Image_Patch_,HR_patch_size)
    
    # MaxDataInput,MinDataInput,LR_Image_Patch=PreProcess_Data(LR_Image_Patch)
    # MaxDataOutput,MinDataOutput,HR_Image_Patch=PreProcess_Data(HR_Image_Patch)
    
    # Predict_Patch=model.predict(LR_Image_Patch)
    # x_decoded=Predict_Patch[:,:,:,0]
    # x_decoded=PostProcess_Data(x_decoded,MaxDataOutput,MinDataOutput)
    
    # rows,cols=my_Im2col.ind2sub(np.array(HR_Image.shape)-HR_patch_size[0]+1,Patch_Idx)
    # IMout=np.zeros(HR_Image.shape)
    # Weight=np.zeros(HR_Image.shape)
    # count=0
    # for index in range(len(cols)):
    #     col=cols[index]
    #     row=rows[index]
    #     block=x_decoded[count,:,:]
    #     IMout[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=block
    #     Weight[row:row+HR_patch_size[0],col:col+HR_patch_size[1]]+=np.ones(HR_patch_size)
    #     count+=1
    # PredictData=IMout/Weight

    FWI_data_true=np.load('OverThrustClip.npy')

    pyplot.figure()
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1870,0]
    gci=pyplot.imshow(FWI_data_true,extent=extent,cmap=cm.seismic,norm=norm)
    
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
    # pyplot.savefig('FWI_True_Model.png',dpi=1000)

    FWI_data=np.load('10Hz_FWI.npy')
    FWI_data=FWI_data.reshape((240,-1))
    FWI_data_FWI=FWI_data[33:-20,20:-20]

    pyplot.figure()
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1870,0]
    gci=pyplot.imshow(FWI_data_FWI,extent=extent,cmap=cm.seismic,norm=norm)
    
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
    # pyplot.savefig('FWI_10Hz.png',dpi=1000)

    FWI_data=np.load('10Hz_LR.npy')
    FWI_data=FWI_data.reshape((90,-1))
    FWI_data_LR=FWI_data[23:-20,20:-20]

    pyplot.figure()
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1880,0]
    gci=pyplot.imshow(FWI_data_LR,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,4000,10)
    y=np.linspace(0,1880,50)
    for idx_x in idx:
        if np.mod(idx_x,40)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)

        
    idx=np.arange(0,1880,10)
    x=np.linspace(0,4000,50)
    for idx_z in idx:
        if np.mod(idx_z,40)==0:
            pyplot.plot(x,[idx_z]*50,'k-',linewidth=0.3)

    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='3%', pad=0.35)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    # pyplot.savefig('FWI_LR_10Hz.png',dpi=1000)
    
    FWI_data_Bicubic=skimage.transform.resize(FWI_data_LR,output_shape=PredictData.shape,mode='symmetric')
    pyplot.figure()
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1880,0]
    gci=pyplot.imshow(FWI_data_Bicubic,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,4000,10)
    y=np.linspace(0,1880,50)
    for idx_x in idx:
        if np.mod(idx_x,40)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        else:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
    idx=np.arange(0,1880,10)
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
    # pyplot.savefig('FWI_Bicubic_10Hz.png',dpi=1000)
    
    
    pyplot.figure()
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1880,0]
    gci=pyplot.imshow(PredictData,extent=extent,cmap=cm.seismic,norm=norm)
    
    idx=np.arange(0,4000,10)
    y=np.linspace(0,1880,50)
    for idx_x in idx:
        if np.mod(idx_x,40)==0:
            pyplot.plot([idx_x,]*50,y,'k-',linewidth=0.3)
        else:
            pyplot.plot([idx_x,]*50,y,'k--',linewidth=0.1)
        
    idx=np.arange(0,1880,10)
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
    # pyplot.savefig('FWI_SRGAN_10Hz.png',dpi=1000)
    
    pyplot.figure()
    index=190
    vp_true_line=FWI_data_true[:,index]
    vp_10Hz_FWI_line=FWI_data_FWI[:,index]
    vp_10Hz_LR_line=FWI_data_LR[:,int(index/4)]
    vp_10Hz_LR_line_=np.zeros(188)
    for idx,data in enumerate(vp_10Hz_LR_line):
        vp_10Hz_LR_line_[4*idx]=data
        vp_10Hz_LR_line_[4*idx+1]=data
        vp_10Hz_LR_line_[4*idx+2]=data
        vp_10Hz_LR_line_[4*idx+3]=data
    vp_10Hz_LR_line=np.zeros(187)
    vp_10Hz_LR_line=vp_10Hz_LR_line_[:-1]

    vp_true_Bicubic_line=FWI_data_Bicubic[:,int(index*2)]
    SRGAN_line=PredictData[:,int(index*2)]

    # pyplot.plot(vp_true_line,np.linspace(0,1870,187),'k--')
    # pyplot.plot(vp_10Hz_FWI_line,np.linspace(0,1870,187),'c-')
    # pyplot.plot(vp_10Hz_LR_line,np.linspace(0,1870,187),'g:')
    # pyplot.plot(vp_true_Bicubic_line[::2],np.linspace(0,1870,187),'b-.')
    # pyplot.plot(SRGAN_line[::2],np.linspace(0,1870,187),'r-')
    
    pyplot.plot(np.linspace(0,1870,187),vp_true_line,'k--')
    pyplot.plot(np.linspace(0,1870,187),vp_10Hz_FWI_line,'c-')
    pyplot.plot(np.linspace(0,1870,187),vp_10Hz_LR_line,'g:')
    pyplot.plot(np.linspace(0,1870,187),vp_true_Bicubic_line[::2],'b-.')
    pyplot.plot(np.linspace(0,1870,187),SRGAN_line[::2],'r-')

    ax=pyplot.gca()
    # ax.invert_yaxis()
    ax.set(aspect=0.3)
    
    pyplot.legend(['True Model','High Resolution','Low Resolution','Bicubic Interpolation','SRGAN'],frameon=False)
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Depth (m)') 
    pyplot.savefig('New_FWI_10Hz_Line.png',dpi=1000)
    