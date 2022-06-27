#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:25:37 2018

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
    frequence=[10,30]
    for idx,freq_ in enumerate(frequence):
        if idx==1:
            continue
        #Model Params
        print(freq_)
        para.xl=101
        para.zl=101
        para.dx=5
        para.dz=5
        para.k_max=1000
        para.dt=5e-4
        #Ricker wavelet main frequence
        para.ricker_freq=freq_
        #Toltal Variation Params
        para.TolVar_key=0
        para.CroGra_key=0
        para.lambda_rho=0.0
        para.lambda_vp=0.0
        para.lambda_CroGra=0.0
        #CPML Params
        para.CPML=12
        para.Npower=2
        para.k_max_CPML=3
        para.alpha_max_CPML=para.ricker_freq*np.pi
        para.Rcoef=1e-8
        #True Model
        rho,vp=Create_Model.Abnormal_Model(para.xl,para.zl,para.CPML)
        # #Source Position
        # x_site=[para.CPML+8]*11+\
        #         [30,40,50,60,70,80,90,100,110,120]+\
        #         [para.CPML+8+para.xl-1]*10+\
        #         [30,40,50,60,70,80,90,100,110]
        # z_site=[20,30,40,50,60,70,80,90,100,110,120]+\
        #         [para.CPML+8+para.zl-1]*10+\
        #         [20,30,40,50,60,70,80,90,100,110]+\
        #         [para.CPML+8]*9
        x_site=[para.CPML+8]*6+\
                [40,60,80,100,120]+\
                [para.CPML+8+para.xl-1]*5+\
                [40,60,80,100]
        z_site=[20,40,60,80,100,120]+\
                [para.CPML+8+para.zl-1]*5+\
                [20,40,60,80,100]+\
                [para.CPML+8]*4
        # x_site=[para.CPML+8]*24
        # z_site=[50]*24
        para.source_site=np.column_stack((x_site,z_site))
        #Receiver Position
        ref_pos_x=[para.CPML+8]*para.zl+\
                  [para.CPML+8+para.xl-1]*para.zl+\
                  list(range(para.CPML+8+1,para.CPML+8+para.xl-1))+\
                  list(range(para.CPML+8+1,para.CPML+8+para.xl-1))
        ref_pos_z=list(range(para.CPML+8,para.CPML+8+para.zl))+\
                  list(range(para.CPML+8,para.CPML+8+para.zl))+\
                  [para.CPML+8]*(para.xl-2)+\
                  [para.CPML+8+para.zl-1]*(para.xl-2)
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
            irho,ivp=Create_Model.Create_Initial_Model(para.xl,para.zl,para.CPML)
        #If the first frequence,Using the last final model
        else:
            dir_path='./%sHz_imodel_file'%frequence[idx-1]
            file_num=int(len(list(Path(dir_path).iterdir()))/2)-1
            data=np.load('./%sHz_imodel_file/%s_imodel.npy'%(frequence[idx-1],file_num))
            irho=data[:int(len(data)/2)].reshape((para.xl+2*8+2*para.CPML,-1))
            ivp=data[int(len(data)/2):].reshape((para.xl+2*8+2*para.CPML,-1))
        #Anonymous function for Gradient Calculate
        fh=lambda x,y:Calculate_Gradient_NoSave_Pool.misfit(x,y,para)    
        
        # # Test Gradient
        # f,g=Calculate_Gradient_NoSave_Pool.misfit(irho,ivp,para)
        # pyplot.figure()
        # pyplot.imshow(g.reshape((para.xl+2*8+2*para.CPML,-1)))
        # pyplot.colorbar()
        
        #Options Params
        options.method='lbfgs'
        options.tol=1e-4
        if idx==0:
            options.maxiter=100
        else:
            options.maxiter=200
        
        Optimization_=Optimization(fh,irho,ivp)
        imodel,info=Optimization_.optimization()
        
        #Plot Vp Data
        pyplot.figure()
        pyplot.imshow(imodel.reshape((para.xl+2*8+2*para.CPML,-1)),cmap=cm.seismic,vmin=1000,vmax=5000)
        pyplot.title('Vp Data')
        pyplot.colorbar()
        #Plot Error Data
        pyplot.figure()
        data_=[]
        for info_ in info:
            data_.append(info_[3])
        pyplot.plot(data_/data_[0])
        pyplot.yscale('log')
    print('Elapsed time is %s seconds !'%str(time.time()-start_time))
