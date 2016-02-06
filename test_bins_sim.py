# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 18:57:21 2015

@author: Atabak
"""


import numpy as np
import theano
import theano.tensor as T
import cPickle
from dA import dA, test


def my_digitize(data,bins):
    result=np.empty(data.shape,dtype='int32')
    for i,g in enumerate(data):
        result[i,:]=np.digitize(g,bins)
    return result
    
normal_scaler=cPickle.load(open('StandardScaler.pkl'))
binary_scaler=cPickle.load(open('MinMaxScaler.pkl'))
inputs_binary, targets_binary, predictions_binary=test()
inputs_normalized = binary_scaler.inverse_transform(inputs_binary)
targets_normalized = binary_scaler.inverse_transform(targets_binary)
predictions_normalized = binary_scaler.inverse_transform(predictions_binary)
inputs_normalized*=3
targets_normalized*=3
predictions_normalized*=3
inputs = normal_scaler.inverse_transform(inputs_normalized)
targets = normal_scaler.inverse_transform(targets_normalized)
predictions = normal_scaler.inverse_transform(predictions_normalized)

bins=[-np.inf,-0.3,-0.1,0.1,0.3,np.inf]
a=13
b=14
targets_bins_11 = np.digitize(targets[:,a],bins)
targets_bins_12 = np.digitize(targets[:,b],bins)
predictions_bins_11 = np.digitize(predictions[:,a],bins)
predictions_bins_12 = np.digitize(predictions[:,b],bins)
accuracy_x=np.mean(targets_bins_11==predictions_bins_11)
print 'accuracy on x axis is = ',accuracy_x
accuracy_y=np.mean(targets_bins_12==predictions_bins_12)
print 'accuracy on y axis is = ',accuracy_y
print 'and on average it is: ', (accuracy_x+accuracy_y)/2

        