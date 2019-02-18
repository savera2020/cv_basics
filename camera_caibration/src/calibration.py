#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 20:19:03 2019

@author: lord
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt





img_points = np.array([[129,273],[889,321],[1649,369],[2433,417],[3233,465],[4067,513],[4913,553],
[193,1041],[971,1097],[1665,1161],[2433,1209],[3217,1273],[4033,1329],[4865,1385],
[665,2097],[1433,2161],[2225,2233],
[329,2393],[1137,2457],
[357,3209],[1409,3305],[2377,3393],[3385,3489],[4417,3609]])


world_points = np.array([[216,72,0],[180,72,0],[144,72,0],[108,72,0],[72,72,0],[36,72,0],[0,72,0],
                [216,36,0],[180,36,0],[144,36,0],[108,36,0],[72,36,0],[36,36,0],[0,36,0],
                [180,0,36],[144,0,36],[108,0,36],
                [180,0,72],[144,0,72],
                [144,0,144],[108,0,144],[72,0,144],[36,0,144],[0,0,144]])




def point_world_matrix(img_point, world_point):
    
    pointX = [world_point[0],world_point[1],world_point[2],1,0,0,0,0,-img_point[0]*world_point[0],-img_point[0]*world_point[1],-img_point[0]*world_point[2],-img_point[0]]
    pointY = [0,0,0,0,world_point[0],world_point[1],world_point[2],1,-img_point[1]*world_point[0],-img_point[1]*world_point[1],-img_point[1]*world_point[2],-img_point[1]]
    return pointX,pointY       ## A is 2 x 12 matrix
    


def DLT_calibration(img_points,world_points):   
    #M = np.empty((2*img_points.shape[0], 12),dtype='int64')
    A = []
    for img_point,world_point in zip(img_points,world_points):
        pointX,pointY = point_world_matrix(img_point, world_point)   ## 2x12 matrix
        
        #np.append(M,np.array(A_point),axis=0)
        A.append(pointX)
        A.append(pointY)
    A = np.array(A)
    
    
    
    ### perfomr SVD 
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    ## use the last value 
    M = np.transpose(vh)
    
    M_1d = M[:,-1]
    M_2d = np.reshape(M_1d,[3,4])
    return M_2d
    
M= DLT_calibration(img_points,world_points)    