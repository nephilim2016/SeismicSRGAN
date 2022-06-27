import numpy as np
from pathlib import Path
from matplotlib import pyplot

if __name__ == '__main__':
    ##################FWI_10Hz##################
    dir_path='./FWI/10Hz_imodel_file'
    d0=np.load('./FWI/10Hz_imodel_file/0_info.npy')
    d0_FWI=d0[3]
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    data_10Hz_num=[]
    data_10Hz_FWI=[]
    for iter_ in range(file_num):
        data=np.load('./FWI/10Hz_imodel_file/%s_info.npy'%iter_)
        data_10Hz_num.append(data[1])
        data_10Hz_FWI.append(data[3]/d0_FWI)
    
    dir_path='./4scale/10Hz_imodel_file'
    d0=np.load('./4scale/10Hz_imodel_file/0_info.npy')
    d0_LR_FWI=d0[3]
    file_num=87
    data_10Hz_LR_num=[]
    data_10Hz_LR_FWI=[]
    for iter_ in range(file_num):
        data=np.load('./4scale/10Hz_imodel_file/%s_info.npy'%iter_)
        data_10Hz_LR_num.append(data[1])
        data_10Hz_LR_FWI.append(data[3]/d0_LR_FWI)
    
    pyplot.figure()
    pyplot.plot(data_10Hz_FWI,'b--')
    pyplot.plot(data_10Hz_LR_FWI,'r-')
    pyplot.legend(['High Resolution','Low Resolution'])
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Data Misfit')
    pyplot.savefig('High_Low.png',dpi=1000)
    
    ###########################################################################
    
    dir_path='./Bicubic_FWI/20Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    data_20Hz_Bicubic_num=[]
    data_20Hz_Bicubic_FWI=[]
    for iter_ in range(file_num):
        data=np.load('./Bicubic_FWI/20Hz_imodel_file/%s_info.npy'%iter_)
        data_20Hz_Bicubic_num.append(data[1])
        data_20Hz_Bicubic_FWI.append(data[3]/d0_LR_FWI)
        
    dir_path='./4scale/20Hz_imodel_file'
    file_num=int(len(list(Path(dir_path).iterdir()))/2)
    data_20Hz_SR_num=[]
    data_20Hz_SR_FWI=[]
    for iter_ in range(file_num):
        data=np.load('./4scale/20Hz_imodel_file/%s_info.npy'%iter_)
        data_20Hz_SR_num.append(data[1])
        data_20Hz_SR_FWI.append(data[3]/d0_LR_FWI)
    
    
    pyplot.figure()
    pyplot.plot(data_20Hz_Bicubic_FWI,'b--')
    pyplot.plot(data_20Hz_SR_FWI,'r-')
    pyplot.legend(['Bicubic Interpolation','SSRGAN'])
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Data Misfit')
    pyplot.savefig('LR_Compare.png',dpi=1000)