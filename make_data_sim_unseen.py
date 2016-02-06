# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 12:35:11 2015

@author: Atabak
"""

# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import theano
import cPickle
from copy import deepcopy
from scipy import io as sp_io


def save_data():
    aff_data_address=r'E:/Projects/Affordances/Aff older data/MATLAB/Data/Seen objects/affDataCont.mat'
    aff_data_dict=sp_io.loadmat(aff_data_address)
    allData=np.asarray(aff_data_dict['rawData'])
    std_scaler=preprocessing.StandardScaler()
    std_scaler=std_scaler.fit(allData)
    with open('StandardScaler.pkl', 'w') as f:
        cPickle.dump(std_scaler, f)
    allData=std_scaler.transform(allData)
    allData/=3.0
    allData[allData>1]=1
    allData[allData<-1]=-1
    scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 0.9))
    scaler = scaler.fit(allData)
    with open('MinMaxScaler.pkl', 'w') as f:
        cPickle.dump(scaler, f)
    result = np.asarray(scaler.transform(allData))
    allData_scaled = shuffle(result, random_state = 3)
    
    
    test_data_address=r'E:/Projects/Affordances/Aff older data/MATLAB/Data/Unseen object/unseenCont.mat'
    aff_data_dict=sp_io.loadmat(test_data_address)
    testData=np.asarray(aff_data_dict['rawData_test'])
    testData=std_scaler.transform(testData)
    testData/=3.0
    testData[testData>1]=1
    testData[testData<-1]=-1
    test_result = np.asarray(scaler.transform(testData))
    testData_scaled = shuffle(test_result, random_state = 3)
    train_data_x=allData_scaled    
    test_data_x=testData_scaled
    

    train_data_y=deepcopy(train_data_x)
    test_data_y=deepcopy(test_data_x)
    
    
    data=train_data_x, test_data_x, train_data_y, test_data_y
    with open('data.pkl', 'w') as f:
        cPickle.dump(data, f)
    
def Load_data(cut=np.inf):

    train_data_x, test_data_x, train_data_y, test_data_y=cPickle.load(open('data.pkl'))
    cut = min(cut,train_data_x.shape[0])
    train_data_x, train_data_y = train_data_x[:cut], train_data_y[:cut]  
    
    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                               borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                               borrow=borrow)
        return shared_x, shared_y
    
    test_data_x = test_data_y.copy()
    test_data_x[:,13:]=0.5    
    test_set_x, test_set_y = shared_dataset(test_data_x, test_data_y)
    train_set_x, train_set_y = shared_dataset(train_data_x, train_data_y)
    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval


if __name__ == '__main__':
    save_data()