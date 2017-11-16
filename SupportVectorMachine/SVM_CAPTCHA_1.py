'''
Created on Nov 13, 2017

@author: yiliu
'''

import numpy as np
from PIL import Image
from sklearn import svm

import load_data

dataset,dataLabel = load_data.batchLoadImage( "/Users/yiliu/Box Sync/Stanford/Classes/2017-2018 Fall/CS 229 Machine Learning/Project/CAPTCHA Database/Single_letter" )
dataLabel = dataLabel.reshape(-1)
multiClassSVM = svm.LinearSVC()
multiClassSVM.fit(dataset,dataLabel)
# print dataLabel
confidence = multiClassSVM.decision_function(dataset[[0],:])
print confidence