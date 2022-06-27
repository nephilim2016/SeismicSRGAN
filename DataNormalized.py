#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:36:47 2020

@author: nephilim
"""
import numpy as np
def DataNormalized(Data):
    MinData=np.min(Data)
    Data-=MinData
    MaxData=np.max(Data)
    Data/=MaxData
    # Data-=1
    return Data

def InverseDataNormalized(Data,NormalData):
    MinData=np.min(NormalData)
    MaxData=np.max(NormalData)
    Factor=MaxData-MinData
    InverseData=(Data+1)*Factor/2+MinData
    return InverseData