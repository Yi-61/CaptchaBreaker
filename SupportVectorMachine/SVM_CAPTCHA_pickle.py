'''
Created on Nov 13, 2017

@author: yiliu
'''

import numpy as np
from PIL import Image
from sklearn import svm

import load_data
import load_pickle_database
from nltk.chunk.util import accuracy

[dataset_read,label_read] = load_pickle_database.load_images_labels("/Users/yiliu/Box Sync/eclipse-workspace/SVM_CAPTCHA/src/Data/1000_single_letter_db.p")
# dataLabel = dataLabel.reshape(-1)
print(dataset_read.shape)
# print(np.nonzero(label_read))
nTrain = 900
trainData = dataset_read[0:nTrain,:]
trainLabel = label_read[0:nTrain,]
multiClassSVM = svm.LinearSVC()
multiClassSVM.fit(trainData,trainLabel)
print(trainData.shape)

testData = dataset_read[nTrain:,:]
testLabel = label_read[nTrain:,]

confidence = multiClassSVM.decision_function(testData)
print(confidence.shape)
# print(confidence[[0],])
prediction = np.argmax(confidence, axis=1)
print(prediction.shape)
print(prediction[[0,],])
#prediction has the shape of (50000,)
comparison = np.equal(prediction,testLabel)
accuracy = sum(comparison)/testLabel.size
print(accuracy)
