# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:33:06 2018

@author: ushasi2
"""
#from graphcnn.helper import *
import scipy.io
import numpy as np
import datetime
import h5py
#import graphcnn.setup.helper
#import graphcnn.setup as setup

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


def load_mspan_dataset():

    dataset = h5py.File('dataset/DSRSID.mat')
    #dataset = dataset['new_dataset']
    #mulim = np.squeeze(dataset['MUL_IMAGES']) #multispectral images
    panim = np.squeeze(dataset['PAN_IMAGES'])  # pan images
    labels = np.squeeze(dataset['LAND_COVER_TYPES'])  #image class number to keep track

    #np.array(mulim) 
    np.array(panim) 
    np.array(labels)
    print("Training set (images) shape: {shape}", panim.shape) 
    mulim = 1
    #print("Training set (labels) shape: {shape}", panim.shape)    
    #loading features in which NaN values have been replaced
      
    # loading multi-labels 
    #labels = scipy.io.loadmat('dataset/LandUse_multilabels.mat')
    #labels = labels['labels']
#labels = np.transpose(labels,(1,0))
    
    # Calculating class weights
   
    return mulim, panim, labels
