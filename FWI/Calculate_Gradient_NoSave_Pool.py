#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:38:43 2018

@author: nephilim
"""
from multiprocessing import Pool,Manager,Process
import numpy as np
import time
import Add_CPML
import Wavelet
import Time_loop
import Reverse_time_loop

def calculate_gradient(rho,vp,index,CPML_Params,para):
    #Get Forward Params
    t=np.arange(para.k_max)*para.dt
    f=Wavelet.ricker(t,para.ricker_freq)
    #True Model Profile Data
    data=para.data[index]
    #Get Forward Data ----> <Generator>
    Forward_data=Time_loop.time_loop(para.xl,para.zl,para.dx,para.dz,para.dt,\
                                                  rho,vp,CPML_Params,f,para.k_max,\
                                                  para.source_site[index],para.ref_pos)
    #Get Generator Data
    V_data=[]
    idata=np.zeros((para.k_max,len(para.ref_pos)))
    for i in range(para.k_max):
        tmp=Forward_data.__next__()
        V_data.append(np.array(tmp[0]))
        idata[i,:]=tmp[1]
    #Get Residual Data
    rhs_data=idata-data
    #Get Reversion Data ----> <Generator>
    Reverse_data=Reverse_time_loop.reverse_time_loop(para.xl,para.zl,para.dx,para.dz,\
                                                     para.dt,rho,vp,CPML_Params,para.k_max,\
                                                     para.ref_pos,rhs_data)
    #Get Generator Data
    RT_P_data=[]
    for i in range(para.k_max):
        tmp=Reverse_data.__next__()
        RT_P_data.append(np.array(tmp[0]))
    RT_P_data.reverse() 
    #Calculate Gradient
    time_sum=np.zeros((para.xl+2*8+2*CPML_Params.CPML,para.zl+2*8+2*CPML_Params.CPML))
    for k in range(1,para.k_max-1):
        u1=V_data[k+1]
        u0=V_data[k-1]
        p1=RT_P_data[k]
        time_sum+=p1*(u1-u0)/para.dt/2
        
    g_vp=2/(rho*vp**3)*time_sum
    return rhs_data.flatten(),g_vp.flatten()    
  

def misfit(rho,vp,para):  
    #Get Toltal Variation Params
    start_time=time.time()
    #Create CPML
    vp_max=max(vp.flatten())
    CPML_Params=Add_CPML.Add_CPML(para.xl,para.zl,para.CPML,vp_max,para.dx,para.dz,para.dt,\
                                  para.Npower,para.k_max_CPML,para.alpha_max_CPML,para.Rcoef)
    #Calculate Gradient
    g_vp=0.0
    rhs=[]
    pool=Pool(processes=8)
    res_l=[]
    for index,value in enumerate(para.source_site):
        res=pool.apply_async(calculate_gradient,args=(rho,vp,index,CPML_Params,para))
        res_l.append(res)
    pool.close()
    pool.join()
    for res in res_l:
        result=res.get()
        rhs.append(result[0])
        g_vp+=result[1]
        del result
    del res_l
    pool.terminate()
    #Get Profile Data
    rhs=np.array(rhs)
    #Get Function Error
    f=0.5*np.linalg.norm(rhs.flatten(),2)**2
    print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
    return f,g_vp


