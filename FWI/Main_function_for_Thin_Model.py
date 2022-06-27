#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 01:23:43 2020

@author: nephilim
"""

import Create_Model
import Forward2D
import Calculate_Gradient_NoSave_Pool
import numpy as np
import time
from pathlib import Path
from Optimization import para,options,Optimization
from matplotlib import pyplot,cm

if __name__=='__main__':       
    start_time=time.time() 
    #Multiscale
    frequence=[10,20]
    for idx,freq_ in enumerate(frequence):
        # if idx==0:
        #     continue
        #Model Params
        print(freq_)
        para.xl=94
        para.zl=200
        para.dx=20
        para.dz=20
        para.k_max=2500
        para.dt=5e-4
        #Ricker wavelet main frequence
        para.ricker_freq=freq_
        #CPML Params
        para.CPML=12
        para.Npower=2
        para.k_max_CPML=3
        para.alpha_max_CPML=para.ricker_freq*np.pi
        para.Rcoef=1e-8
        #True Model
        rho=2000
        vp=Create_Model.Overthrust_Model_half(para.xl,para.zl,para.CPML)
        # #Source Position
        # x_site=[para.CPML+8]*11+\
        #         [30,40,50,60,70,80,90,100,110,120]+\
        #         [para.CPML+8+para.xl-1]*10+\
        #         [30,40,50,60,70,80,90,100,110]
        # z_site=[20,30,40,50,60,70,80,90,100,110,120]+\
        #         [para.CPML+8+para.zl-1]*10+\
        #         [20,30,40,50,60,70,80,90,100,110]+\
        #         [para.CPML+8]*9
        x_site=[para.CPML+8]*21
        z_site=[20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220]
        # x_site=[para.CPML+8]*2
        # z_site=[120]*2
        para.source_site=np.column_stack((x_site,z_site))
        #Receiver Position
        ref_pos_x=[para.CPML+8]*para.zl
        ref_pos_z=list(range(para.CPML+8,para.CPML+8+para.zl))
        para.ref_pos=np.column_stack((ref_pos_x,ref_pos_z))    
        #Get True Model Data
        Forward2D.Forward_2D(rho,vp,para)
        print('Forward Done !')
        print('Elapsed time is %s seconds !'%str(time.time()-start_time))
        #Save Profile Data
        para.data=[]
        for i in range(len(para.source_site)):
            data=np.load('./%sHz_forward_data_file/%sx_%sz_record.npy'%(para.ricker_freq,para.source_site[i][0],para.source_site[i][1]))
            para.data.append(data)
        #If the first frequence,Create Initial Model
        if idx==0:
            ivp=Create_Model.Initial_Overthrust_Model(vp,40)
            irho=2000
        #If the first frequence,Using the last final model
        else:
            dir_path='./%sHz_imodel_file'%frequence[idx-1]
            file_num=int(len(list(Path(dir_path).iterdir()))/2)-1
            data=np.load('./%sHz_imodel_file/%s_imodel.npy'%(frequence[idx-1],file_num))
            irho=2000
            ivp=data.reshape((para.xl+2*8+2*para.CPML,-1))
        #Anonymous function for Gradient Calculate
        fh=lambda x,y:Calculate_Gradient_NoSave_Pool.misfit(x,y,para)    
        
        # # Test Gradient
        # f,g=Calculate_Gradient_NoSave_Pool.misfit(irho,ivp,para)
        # pyplot.figure()
        # pyplot.imshow(g[:int(len(g)/2)].reshape((para.xl+2*8+2*para.CPML,-1)))
        # pyplot.colorbar()
        # pyplot.figure()
        # pyplot.imshow(g[int(len(g)/2):].reshape((para.xl+2*8+2*para.CPML,-1)))
        # pyplot.colorbar()
        
        #Options Params
        options.method='lbfgs'
        options.tol=1e-4
        options.maxiter=200
        Optimization_=Optimization(fh,irho,ivp)
        imodel,info=Optimization_.optimization()
        
        #Plot Vp Data
        pyplot.figure()
        pyplot.imshow(imodel.reshape((para.xl+2*8+2*para.CPML,-1)),cmap=cm.seismic,vmin=2000,vmax=6000)
        pyplot.title('Vp Data')
        pyplot.colorbar()
        # #Plot Error Data
        # pyplot.figure()
        # data_=[]
        # for info_ in info:
        #     data_.append(info_[3])
        # pyplot.plot(data_/data_[0])
        # pyplot.yscale('log')
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))
