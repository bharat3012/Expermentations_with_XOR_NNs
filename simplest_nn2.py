#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 17:33:33 2018

@author: bharat
"""

import numpy as np
import time

n_hidden = 10

#Distance_Train,speed_Train,Distance_car, Speed_car,Proximity
n_in = 5
n_out =5
n_samples = 300

#Hyperparameters
learning_rate = 0.01
momentum = 0.9

#non determininstic seed
np.random.seed(0)

#activation function

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

    #for back prop
def tanh_prime(x):
    return  1 - np.tanh(x)**2

#training--input data, transpose, layer1, layer2, biases
def train(x, t, V, W, bv, bw):

    # forward
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    # backward
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)
    
    #predict loss
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)
    #cross entropy
    loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) )

    # Note that we use error for each layer as a gradient
    # for biases

    return  loss, (dV, dW, Ev, Ew)

def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

# Setup initial parameters
# Note that initialization is cruxial for first-order methods!
    #2 layered 
#weights
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

#initialize biases
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

#parameters
params = [V,W,bv,bw]

# Generate some data
#300 samples generations
X = np.random.binomial(1, 0.5, (n_samples, n_in))
T = X ^ 1

# Last step Training time 
for epoch in range(100):
    err = []
    upd = [0]*len(params)

    t0 = time.clock()
    #for each data point, update weights
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], *params)
        #update loss
        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]

        err.append( loss )

    print ("Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 ))

# Try to predict something

x = np.random.binomial(1, 0.5, n_in)
print ("XOR prediction:")
print (x)
print (predict(x, *params))