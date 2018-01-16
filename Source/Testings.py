# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:03:20 2017

@author: dbn
"""
#import theano

import numpy as np 

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math
import tensorflow as tf
import theano
import scipy.io as sio
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation

#VARIABLES
CONV_LAYLENGTH = 5
NUM_CONV_FILTS = 32

def loadMLII_arrythmiaData():
    data = []
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\100m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\101m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\103m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\105m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\106m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\107m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\108m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\109m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\111m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\112m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\113m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\114m.mat').get('val')[1])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\115m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\116m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\117m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\118m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\119m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\121m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\122m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\123m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\124m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\200m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\201m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\202m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\203m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\205m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\207m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\208m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\209m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\210m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\212m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\213m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\214m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\215m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\217m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\219m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\220m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\221m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\222m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\223m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\228m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\230m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\231m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\232m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\233m.mat').get('val')[0])
    data.append(sio.loadmat('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\234m.mat').get('val')[0])
    return data

def windowData(data,windowSize):
    chunks = [data[x:x+windowSize] for x in range(0, len(data), windowSize) if len(data[x:x+windowSize])>=windowSize]
    return chunks
def normalise(X):
    Z = np.array([(y -np.min(X))/(np.max(X)-np.min(X)) for y in X])
    return Z

def convFilterVisualiation(model,layer,path,numX,numY):
    x = model.layers[layer].get_weights()
    xt = np.transpose(x[0])
    fig = plt.figure(figsize=(numX*10,numY*10))
    for i in range(0,np.shape(xt)[0]):
        ax = plt.subplot(numX,numY,i+1)
        ax.set_title('Filter '+str(i))
        ax.plot(xt[i][0])
    plt.savefig(path)
    fig.show()
def ConvolutionalAutoEncoder(chunkSize,CFL1=CONV_LAYLENGTH,CFL2=CONV_LAYLENGTH,CFL3=CONV_LAYLENGTH,CFL4=CONV_LAYLENGTH,CFL5=CONV_LAYLENGTH,CFL6=CONV_LAYLENGTH,CFL7=CONV_LAYLENGTH,numConvFilts1 = NUM_CONV_FILTS,numConvFilts2 = NUM_CONV_FILTS,numConvFilts3 = NUM_CONV_FILTS,numConvFilts4 = NUM_CONV_FILTS,numConvFilts5 = NUM_CONV_FILTS,numConvFilts6 = NUM_CONV_FILTS):
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
    from keras.models import Model    
    input_seq = Input(shape=(chunkSize,1, ))  # 3 sec of 300Hz measurements
    hidden1 = Conv1D(numConvFilts1, CFL1, activation='relu', padding='same')(input_seq)
    pool1 = MaxPooling1D(5, padding='same')(hidden1)
    hidden2 = Conv1D(numConvFilts2, CFL2, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(5, padding='same')(hidden2)
    hidden3 = Conv1D(numConvFilts3, CFL3, activation='relu', padding='same')(pool2)
    encoded = MaxPooling1D(3, padding='same')(hidden3)
    
    hidden4 = Conv1D(numConvFilts4, CFL4, activation='relu', padding='same')(encoded)
    pool4 = UpSampling1D(3)(hidden4)
    hidden5 = Conv1D(numConvFilts5, CFL5, activation='relu', padding='same')(pool4)
    pool5 = UpSampling1D(5)(hidden5)
    hidden6 = Conv1D(numConvFilts6, CFL6, activation='relu', padding='same')(pool5)
    pool6 = UpSampling1D(5)(hidden6)
    decoded = Conv1D(1, CFL7, activation='tanh', padding='same')(pool6)
    
    autoencoder = Model(input_seq, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    print(autoencoder.summary())
    return autoencoder,hidden1,hidden2,hidden3,hidden4,hidden5,hidden6,encoded,decoded

def Experiments(data,frequency = 300, amountSeconds = 1,numEpochs = 100,CFL1=CONV_LAYLENGTH,CFL2=CONV_LAYLENGTH,CFL3=CONV_LAYLENGTH,CFL4=CONV_LAYLENGTH,CFL5=CONV_LAYLENGTH,CFL6=CONV_LAYLENGTH,CFL7=CONV_LAYLENGTH,numConvFilts1 = NUM_CONV_FILTS,numConvFilts2 = NUM_CONV_FILTS,numConvFilts3 = NUM_CONV_FILTS,numConvFilts4 = NUM_CONV_FILTS,numConvFilts5 = NUM_CONV_FILTS,numConvFilts6 = NUM_CONV_FILTS,filename = '',numX=10,numY=10):
    #reading the data
    #filestrings = glob.glob('C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\Data\\MIT_BIH Arrhythmia Database\\Matlab\\*.mat')
    #labels =  pd.read_csv(ROOT+ '/Data/training2017/labels.csv',header=None)
    chunkSize =  frequency*amountSeconds
    AE,hidden1,hidden2,hidden3,hidden4,hidden5,hidden6,encoded,decoded = ConvolutionalAutoEncoder(chunkSize,CFL1,CFL2,CFL3,CFL4,CFL5,CFL6,CFL7,numConvFilts1,numConvFilts2,numConvFilts3,numConvFilts4,numConvFilts5,numConvFilts6)
    for i in range(0,len(data)):
        chunkedData = windowData(data[i],chunkSize)
        #normalize each chunk
        chunkedData = np.array([normalise(X) for X in chunkedData])
        #chunkedData = np.array([X.reshape(1,chunkSize,1) for X in chunkedData])
        
            
        if len(chunkedData)==len(chunkedData):
            print('equal lengths')
            Xtrain, Xtest, ytrain, ytest = train_test_split(chunkedData,chunkedData,test_size = 0.3)
       
        AE.fit(Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1],1),Xtrain.reshape(Xtrain.shape[0],Xtrain.shape[1],1),epochs=numEpochs)
    results = AE.predict(Xtest.reshape(Xtest.shape[0],Xtest.shape[1],1))
    plotrange = 20
    for i in range(0,plotrange):
       plt.figure()
       plt.plot(Xtest[i])
       plt.plot(results[i])
       plt.show()
       
    convFilterVisualiation(AE,1,'C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\images\\'+filename+'_Layer1.png',numX,numY)
    convFilterVisualiation(AE,3,'C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\images\\'+filename+'_Layer3.png',numX,numY)
    convFilterVisualiation(AE,5,'C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\images\\'+filename+'_Layer5.png',numX,numY)
    convFilterVisualiation(AE,7,'C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\images\\'+filename+'_Layer7.png',numX,numY)
    convFilterVisualiation(AE,9,'C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\images\\'+filename+'_Layer9.png',numX,numY)
    convFilterVisualiation(AE,11,'C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\images\\'+filename+'_Layer11.png',numX,numY)
    convFilterVisualiation(AE,13,'C:\\Users\\Dirk\\Desktop\\DeepLearningForSignals\\images\\'+filename+'_Layer13.png',numX,numY)    
           
data = loadMLII_arrythmiaData()
frequency = 300
amountSeconds = 1
numEpochs = 100
#Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =5,CFL2 =5,CFL3 =5,CFL4 =5,CFL5 =5,CFL6 =5,CFL7 =5,numConvFilts1= 25,numConvFilts2= 25,numConvFilts3= 25,numConvFilts4= 25,numConvFilts5= 25,numConvFilts6= 25,filename = 'CFL5_NumFilts25',numX = 5,numY =5)
#Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =5,CFL2 =5,CFL3 =5,CFL4 =5,CFL5 =5,CFL6 =5,CFL7 =5,numConvFilts1= 5,numConvFilts2= 5,numConvFilts3= 5,numConvFilts4= 5,numConvFilts5= 5,numConvFilts6= 5,filename = 'CFL5_NumFilts5',numX = 5,numY =1)
#Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =5,CFL2 =5,CFL3 =5,CFL4 =5,CFL5 =5,CFL6 =5,CFL7 =5,numConvFilts1= 10,numConvFilts2= 10,numConvFilts3= 10,numConvFilts4= 10,numConvFilts5= 10,numConvFilts6= 10,filename = 'CFL5_NumFilts10',numX = 5,numY =2)
#Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =25,CFL2 =25,CFL3 =25,CFL4 =25,CFL5 =25,CFL6 =25,CFL7 =25,numConvFilts1= 25,numConvFilts2= 25,numConvFilts3= 25,numConvFilts4= 25,numConvFilts5= 25,numConvFilts6= 25,filename = 'CFL25_NumFilts25',numX = 5,numY =5)
#Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =25,CFL2 =25,CFL3 =25,CFL4 =25,CFL5 =25,CFL6 =25,CFL7 =25,numConvFilts1= 5,numConvFilts2= 5,numConvFilts3= 5,numConvFilts4= 5,numConvFilts5= 5,numConvFilts6= 5,filename = 'CFL25_NumFilts5',numX = 5,numY =1)
#Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =25,CFL2 =25,CFL3 =25,CFL4 =25,CFL5 =25,CFL6 =25,CFL7 =25,numConvFilts1= 10,numConvFilts2= 10,numConvFilts3= 10,numConvFilts4= 10,numConvFilts5= 10,numConvFilts6= 10,filename = 'CFL25_NumFilts10',numX = 5,numY =2)
#Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =100,CFL2 =100,CFL3 =100,CFL4 =100,CFL5 =100,CFL6 =100,CFL7 =100,numConvFilts1= 25,numConvFilts2= 25,numConvFilts3= 25,numConvFilts4= 25,numConvFilts5= 25,numConvFilts6= 25,filename = 'CFL100_NumFilts25',numX = 5,numY =5)
#Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =100,CFL2 =100,CFL3 =100,CFL4 =100,CFL5 =100,CFL6 =100,CFL7 =100,numConvFilts1= 5,numConvFilts2= 5,numConvFilts3= 5,numConvFilts4= 5,numConvFilts5= 5,numConvFilts6= 5,filename = 'CFL100_NumFilts5',numX = 5,numY =1)
#Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =100,CFL2 =100,CFL3 =100,CFL4 =100,CFL5 =100,CFL6 =100,CFL7 =100,numConvFilts1= 10,numConvFilts2= 10,numConvFilts3= 10,numConvFilts4= 10,numConvFilts5= 10,numConvFilts6= 10,filename = 'CFL100_NumFilts10',numX = 5,numY =2)

Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =5,CFL2 =5,CFL3 =5,CFL4 =5,CFL5 =5,CFL6 =5,CFL7 =5,numConvFilts1= 1,numConvFilts2= 1,numConvFilts3= 1,numConvFilts4= 1,numConvFilts5= 1,numConvFilts6= 1,filename = 'CFL5_NumFilts1',numX = 1,numY =1)
Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =5,CFL2 =5,CFL3 =5,CFL4 =5,CFL5 =5,CFL6 =5,CFL7 =5,numConvFilts1= 300,numConvFilts2= 300,numConvFilts3= 300,numConvFilts4= 300,numConvFilts5= 300,numConvFilts6= 300,filename = 'CFL5_NumFilts300',numX = 15,numY =20)
Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =25,CFL2 =25,CFL3 =25,CFL4 =25,CFL5 =25,CFL6 =25,CFL7 =25,numConvFilts1= 1,numConvFilts2= 1,numConvFilts3= 1,numConvFilts4= 1,numConvFilts5= 1,numConvFilts6= 1,filename = 'CFL25_NumFilts1',numX = 1,numY =1)
Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =25,CFL2 =25,CFL3 =25,CFL4 =25,CFL5 =25,CFL6 =25,CFL7 =25,numConvFilts1= 300,numConvFilts2= 300,numConvFilts3= 300,numConvFilts4= 300,numConvFilts5= 300,numConvFilts6= 300,filename = 'CFL25_NumFilts300',numX = 15,numY =20)
Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =100,CFL2 =100,CFL3 =100,CFL4 =100,CFL5 =100,CFL6 =100,CFL7 =100,numConvFilts1= 1,numConvFilts2= 1,numConvFilts3= 1,numConvFilts4= 1,numConvFilts5= 1,numConvFilts6= 1,filename = 'CFL100_NumFilts1',numX = 1,numY =1)
Experiments(data,frequency = frequency,amountSeconds = amountSeconds,numEpochs = numEpochs,CFL1 =100,CFL2 =100,CFL3 =100,CFL4 =100,CFL5 =100,CFL6 =100,CFL7 =100,numConvFilts1= 300,numConvFilts2= 300,numConvFilts3= 300,numConvFilts4= 300,numConvFilts5= 300,numConvFilts6= 300,filename = 'CFL100_NumFilts300',numX = 15,numY =20)
#from keras import backend as K
#
#inp = AE.input                                           # input placeholder
#outputs = [layer.output for layer in AE.layers]          # all layer outputs
#functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
#
## Testing
#test = Xtest.reshape(Xtest.shape[0],Xtest.shape[1],1)
#layer_outs = [func([test, 1.]) for func in functors]
#plt.plot(layer_outs[3][0][0])
##[number layer][always 0][testsample number][node number in layer]
#from scipy.interpolate import interp1d
#
##change first[] to change the layer to disp
##layer = 7
##for node in range(0,np.shape(layer_outs[layer][0][0])[1]):
##    x = np.linspace(0,np.shape(layer_outs[layer][0][0][node])[0]-1,num=np.shape(layer_outs[layer][0][0][node])[0],endpoint=True)
##    y = layer_outs[layer][0][0][node]
##    f = interp1d(x,y,kind = 'cubic')
##    xnew = np.linspace(0,np.shape(layer_outs[layer][0][0][node])[0]-1,num = np.shape(layer_outs[0][0][0])[0],endpoint=True)
##    plt.figure();plt.plot(Xtest[0]);plt.plot(f(xnew));plt.show
#
#
#
#def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
#    print('----- activations -----')
#    activations = []
#    inp = model.input
#
#    model_multi_inputs_cond = True
#    if not isinstance(inp, list):
#        # only one input! let's wrap it in a list.
#        inp = [inp]
#        model_multi_inputs_cond = False
#
#    outputs = [layer.output for layer in model.layers if
#               layer.name == layer_name or layer_name is None]  # all layer outputs
#
#    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
#
#    if model_multi_inputs_cond:
#        list_inputs = []
#        list_inputs.extend(model_inputs)
#        list_inputs.append(0.)
#    else:
#        list_inputs = [model_inputs, 0.]
#
#    # Learning phase. 0 = Test mode (no dropout or batch normalization)
#    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
#    layer_outputs = [funcs(list_inputs)[0] ]
#    for layer_activations in layer_outputs:
#        activations.append(layer_activations)
#        if print_shape_only:
#            print(layer_activations.shape)
#        else:
#            print(layer_activations)
#    return activations
#
#
#def display_activations(activation_maps,disp1Example=True):
#    for layer in range(0,14):
#        if disp1Example:
#            for ex in range(0,1):
#                print('filters of layer '+str(layer)+' example '+str(ex))
#                plt.figure()
#                plt.plot(a[layer][ex])
#                plt.show()
#        else:
#            for ex in range(0,np.shape(activation_maps[layer])[0]):
#                print('filters of layer '+str(layer)+' example '+str(ex))
#                plt.figure()
#                plt.plot(a[layer][ex])
#                plt.show()
#        print('displaying filters')
#    import seaborn as sb
#    sb.set()
#    for i in range(0,14):
#        plt.figure()
#        sb.heatmap(np.transpose(a[i][0]))
#        plt.show()
#        
#test = Xtest.reshape(Xtest.shape[0],Xtest.shape[1],1)
#a = get_activations(AE,test, print_shape_only=True)  # with just one sample.
#display_activations(a)