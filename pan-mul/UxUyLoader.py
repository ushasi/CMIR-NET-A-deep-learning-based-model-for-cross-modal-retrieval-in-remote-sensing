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


def load_uxuy_dataset():

    datasetX = scipy.io.loadmat('dataset/mul_features.mat')
    mulim = np.squeeze(datasetX['features'])  # mul images
    datasetY = scipy.io.loadmat('dataset/pan_features.mat')
    panim = np.squeeze(datasetY['features'])  # pan images
    labels = scipy.io.loadmat('dataset/labels.mat')
    labels = np.squeeze(labels['LAND_COVER_TYPES'])  #image class number to keep track

    np.array(mulim) 
    np.array(panim) 
    np.array(labels)
    print("Training set (images X) shape: {shape}", panim.shape)
    print("Training set (images Y) shape: {shape}", mulim.shape) 
    print("Training set (labels) shape: {shape}", labels.shape)    
    #loading features in which NaN values have been replaced
      
    # loading multi-labels 
    #labels = scipy.io.loadmat('dataset/LandUse_multilabels.mat')
    #labels = labels['labels']
#labels = np.transpose(labels,(1,0))
    
    # Calculating class weights
   
    return mulim, panim, labels
