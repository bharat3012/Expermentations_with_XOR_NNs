#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:40:53 2018

@author: bharat
"""

import numpy as np


def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input data
X= np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])

y= np.array([[0],
             [1],
             [1],
             [0]])
    
#let say 1 output neuron each
np.random.seed(1)

#Create synapses--As their are 3 layers we need 2 connections(synapses)
#aSSIGN RANDOM WEIGHTS TO THE SYNAPES matrices
syn0 =2*np.random.random((3,4))-1    
syn1 = 2*np.random.random((4,1))-1

#Start training code
for j in range(60000):
    
    l0 = X 
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
 
#Error    
    l2_error =y-l2
#Iterate multiple times    
    if(j%10000)==0:
        print("Error:" +str(np.mean(np.abs(l2_error))))
    #derivative of layer 2    
    l2_delta = l2_error*nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)  
    
    #derivative of layer 1
    l1_delta = l1_error*nonlin(l1, deriv=True)
    
    #Backpropagation---Update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l2.T.dot(l1_delta)
    
print("Output after training")
print(l2)    
    