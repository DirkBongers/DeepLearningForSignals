# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:03:20 2017

@author: dbn
"""
#import theano
import tensorflow as tf
import scipy.io as sio
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation
ROOT = 'C:/Users/dbn/Desktop/DeepLearningForSignals'
data = []
filestrings = glob.glob(ROOT + '/Data/training2017/*.mat')
labels =  pd.read_csv(ROOT+ '/Data/training2017/labels.csv',header=None)
for i in range(0,len(filestrings)):
    data.append(sio.loadmat(filestrings[i]).get('val'))
    data[i] = data[i][0]
plotrange = 100
for i in range(0,len(data[0:plotrange])):    
    plt.figure(i)
    plt.plot(data[i])
    
if len(labels)==len(data):
    print('equal lengths')
    Xtrain, Xtest, ytrain, ytest = train_test_split(data,labels,test_size = 0.3)
    
#    model = Sequential()
#    model.add(Dense(units = 64,input_dim =len(Xtrain)))
#    model.add(Activation('relu'))
#    model.add(Dense(units = 10))
#    model.add(Activation('softmax'))
#    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics = ['accuracy'])
#    model.fit(np.array(Xtrain),np.array(ytrain),epochs=5,batch_size=32)
#    loss_and_metrics = model.evaluate(Xtest,ytest,batch_size=128)
# 