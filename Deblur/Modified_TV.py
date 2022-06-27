# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:09:22 2018

@author: Yilin Liu
"""
import numpy as np
from numba import jit

def denoising_2D_TV(data):
    data_max=np.max(data)
    data=data/data_max
    M,N=np.shape(data)
    X0=np.zeros((M+2,N+2))
    X0[1: M+1,1: N+1]=data
    Y0=np.zeros((M+2,N+2))
    Y0[1: M+1,1: N+1]=data
    X=np.zeros((M+2,N+2))
    Zx=np.zeros((M+2,N+2))
    Zy=np.zeros((M+2,N+2))
    Ux=np.zeros((M+2,N+2))
    Uy=np.zeros((M+2,N+2))
    lamda=0.01
    rho_=1
    num=1000
    err=1e-6
    return_data=denoising_2D_TV_(num,X,X0,err,M,N,Zx,Zy,Ux,Uy,Y0,lamda,rho_)
    return return_data*data_max

@jit(nopython=True)
def denoising_2D_TV_(num,X,X0,err,M,N,Zx,Zy,Ux,Uy,Y0,lamda,rho_):
    K=0
    while K<num and np.linalg.norm(X-X0,2) > err:
        # update X
        X0=X
        MM=M+2
        NN=N+2
        D=np.zeros((MM,NN))
        D[:,0:NN-1]=Zx[:,0:NN-1]-Zx[:,1:NN]
        D[:,NN-1]=Zx[:,NN-1]-Zx[:,0]
        Dxt_Zx=D
        
        D=np.zeros((MM,NN))
        D[0:MM-1,:]=Zy[0:MM-1,:]-Zy[1:MM,:]
        D[MM-1,:]=Zy[MM-1,:]-Zy[0,:]
        Dyt_Zy=D
        
        D=np.zeros((MM,NN))
        D[:,0:NN-1]=Ux[:,0:NN-1]-Ux[:,1:NN]
        D[:,NN-1]=Ux[:,NN-1]-Ux[:,0]
        Dxt_Ux=D
        
        D=np.zeros((MM,NN))
        D[0:MM-1,:]=Uy[0:MM-1,:]-Uy[1:MM,:]
        D[MM-1,:]=Uy[MM-1,:]-Uy[0,:]
        Dyt_Uy=D
        
        RHS=Y0+lamda*rho_*(Dxt_Zx+Dyt_Zy)-lamda*(Dxt_Ux+Dyt_Uy)
        X=np.zeros((M+2,N+2))
        
        for i in range(1,M+1):
            for j in range(1,N+1):
                X[i,j]=((X0[i+1,j]+X0[i-1,j]+X0[i,j+1]+X0[i,j-1])*lamda*rho_+RHS[i,j])/(1+4*lamda*rho_)
                
        # update Z

        D=np.zeros((MM,NN))
        D[:,1:NN]=X[:,1:NN]-X[:,0:NN-1]
        D[:,0]=X[:,0]-X[:,NN-1]
        Dx_X=D
        D=np.zeros((MM,NN))
        D[1:MM,:]=X[1:MM,:]-X[0:MM-1,:]
        D[0,:]=X[0,:]-X[MM-1,:]
        Dy_X=D
        Tx=Ux/rho_+Dx_X
        Ty=Uy/rho_+Dy_X
        
        
        Zx=np.fmax(np.fabs(Tx)-1/rho_,0)*np.sign(Tx)
        Zy=np.fmax(np.fabs(Ty)-1/rho_,0)*np.sign(Ty)
        
        # update U
        Ux=Ux+(Dx_X-Zx)
        Uy=Uy+(Dy_X-Zy)
        K+=1
    print(K)
    print(np.linalg.norm(X-X0,2))
    return X[1:M+1,1:N+1]

if __name__=='__main__':
    Deblur=denoising_2D_TV(PredictData)
    
    pyplot.figure(1)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1880,0]
    gci=pyplot.imshow(Blur_Image,cmap=cm.seismic,extent=extent,norm=norm)
    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='5%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    # # pyplot.savefig('sigma_%s_Result.png'%sigma,dpi=1000)
    
    pyplot.figure(2)
    norm=colors.Normalize(vmin=2500, vmax=6000)
    extent=[0,4000,1880,0]
    gci=pyplot.imshow(Deblur,cmap=cm.seismic,extent=extent,norm=norm)
    ax=pyplot.gca()
    divider=make_axes_locatable(ax)
    cax=divider.append_axes('right', size='5%', pad=0.15)
    cbar=pyplot.colorbar(gci,cax=cax)
    cbar.set_label('$m/s$')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Depth (m)') 
    
    pyplot.figure(3)
    index=190
    vp_true_line=Clean_Image[:,index]
    vp_blur_line=Blur_Image[:,index]
    vp_deblur_line=Deblur[:,index]
    pyplot.plot(vp_true_line,np.linspace(0,1870,187),'k--')
    pyplot.plot(vp_blur_line,np.linspace(0,1870,187),'b-.')
    pyplot.plot(vp_deblur_line,np.linspace(0,1870,187),'r-')
    ax=pyplot.gca()
    ax.invert_yaxis()
    ax.set(aspect=3)
    pyplot.legend(['True Model','Low Resolution','SRE-ResNet'],frameon=False)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Depth (m)') 