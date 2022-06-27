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
    Data/=(MaxData/2+1e-6)
    Data-=1
    return Data,MaxData,MinData


def InverseDataNormalized(Data,MaxData,MinData):
    Data+=1
    Data*=(MaxData/2+1e-6)
    Data+=MinData
    return Data