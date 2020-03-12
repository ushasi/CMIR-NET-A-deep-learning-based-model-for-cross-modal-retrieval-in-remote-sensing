# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:33:06 2018

@author: ushasi2
"""
#from graphcnn.helper import *
import scipy.io
import numpy as np
import datetime
#import word2veccode as w2v

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"



def load_imgspeech_dataset():

    dataset = scipy.io.loadmat('dataset/scores.mat')
    #dataset = dataset['new_dataset']
    scores = np.squeeze(dataset['scores'])  # cnn features image
    dataset = scipy.io.loadmat('dataset/speech.mat')
    vggish = np.squeeze(dataset['features'])  # word2vec features multi-label

    np.array(vggish) 
    np.array(scores) 
    #np.array(labels)
    print("Training set (images) shape: {shape}", scores.shape) 
    
    #words = w2v.load_word2vec_dataset()
    print("Training set (words) shape: {shape}", vggish.shape)
        
      
    # loading multi-labels 
    labels = scipy.io.loadmat('dataset/LandUse_multilabels.mat')
    labels = labels['labels']
    labels = np.transpose(labels,(1,0))

    dataset = scipy.io.loadmat('dataset/splab.mat')
    splab =  np.squeeze(dataset['splab'])
    
    # Calculating class weights
   
    return scores, vggish, labels, splab
