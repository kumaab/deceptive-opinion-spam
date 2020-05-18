#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os,fnmatch
import numpy as np
import pandas as pd
import copy
import itertools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
stop_words = set(stopwords.words('english'))
np.set_printoptions(threshold=sys.maxsize)
####################################
# Metrics
####################################
def result(confusionM):
    print('----------TRUTHFUL---------')
    PrecisionT = confusionM[0][0]/(confusionM[0][0] + confusionM[0][1])
    RecallT = confusionM[0][0]/(confusionM[0][0] + confusionM[0][2])
    FScore = 2*PrecisionT*RecallT/(PrecisionT + RecallT)
    print(pd.DataFrame({'P':[PrecisionT],'R':[RecallT],'F':[FScore]}).to_string(index=False))
    print('----------DECEPTIVE---------')
    PrecisionT = confusionM[1][0]/(confusionM[1][0] + confusionM[1][1])
    RecallT = confusionM[1][0]/(confusionM[1][0] + confusionM[1][2])
    FScore = 2*PrecisionT*RecallT/(PrecisionT + RecallT)
    print(pd.DataFrame({'P':[PrecisionT],'R':[RecallT],'F':[FScore]}).to_string(index=False))
    return

###################################
# Confusion Matrix
###################################
def confusion(predicted,actual):
    n = len(predicted)
    labels = [0,1]
    # Decision Matrix : TruePositives(TP), FalsePositives(FP), FalseNegatives(FN)
    matrix = [[0,0,0] for l in labels]
    for l in labels:
        #TruePositive
        matrix[l][0] = sum(1 for i in range(n) if predicted[i] == actual[i] == l)
        #FalsePositive
        matrix[l][1] = sum(1 for i in range(n) if predicted[i] == l and predicted[i] != actual[i])
        #FalseNegative
        matrix[l][2] = sum(1 for i in range(n) if actual[i] == l and predicted[i] != actual[i])
    return matrix

#################################
# 5 Fold Nested Cross Validation
#################################
def nested_cv(shuffle=False):
    labels = [0,1]
    # hyperparameter values to optimize over
    grid = {'C': [1, 10, 100]}
    CM = np.array([[0,0,0] for l in labels])
    ACC = []
    for i in range(5):
        order = [j for j in range(5) if j != i]
        order.append(i)
        X,Y = tfidf_unigrams(order)
        trainlen = 4*X.shape[0]//5
        trainX,testX = copy.deepcopy(X[:trainlen]),copy.deepcopy(X[trainlen:])
        trainY,testY  = copy.deepcopy(Y[:trainlen]),copy.deepcopy(Y[trainlen:])
        if shuffle:
            trainX, trainY = shuffle(trainX, trainY, random_state=34)
        keys, values = zip(*grid.items())
        bestC,bestAcc = -1,-1
        for c in grid['C']:# try all combinations of hyperparameters
            fold,size,acc = 0,(trainX.shape[0]//5),[]
            #########################################
            # This is where the MAGIC happens
            #########################################
            #Linear Kernel
            svm = SVC(C=c,kernel='linear')
            # Inner CrossValidation for HyperParameter Tuning
            for j in range(5):
                innertestX = trainX[fold: fold + size]
                innertrainX = np.concatenate([trainX[:fold],trainX[fold+size:]])
                innertestY = trainY[fold: fold + size]
                innertrainY = np.concatenate([trainY[:fold],trainY[fold+size:]])
                svm.fit(innertrainX, innertrainY)
                prediction = svm.predict(innertestX)
                acc.append(metrics.accuracy_score(innertestY, prediction))
                fold += (size)
            if np.mean(acc) > bestAcc:
                bestC = c
        # Now Run SVM AGAIN on the BEST Hyperparamters
        svm = SVC(C=bestC,kernel='linear')
        svm.fit(trainX, trainY)
        prediction = svm.predict(testX)
        ACC.append(metrics.accuracy_score(testY, prediction))
        CM += confusion(prediction,testY)
    print('Accuracy : ',round(np.mean(ACC),2))
    result(CM)
    return

def tfidf_unigrams(order):
    corpus,Y = [],[]
    dataset = [ #TRUTHFUL
                ['data/TripAdvisor/fold1',
                 'data/TripAdvisor/fold2',
                 'data/TripAdvisor/fold3',
                 'data/TripAdvisor/fold4',
                 'data/TripAdvisor/fold5',
                ],
                #DECEPTIVE
                ['data/MTurk/fold1',
                 'data/MTurk/fold2',
                 'data/MTurk/fold3',
                 'data/MTurk/fold4',
                 'data/MTurk/fold5',
                ]
              ]
    for i in order:
        for d in range(2):
            prefix = dataset[d][i]
            for fname in os.listdir(prefix):
                if fnmatch.fnmatch(fname, '*.uni.txt'):
                    file = open(prefix + '/' + fname,"r")
                    data = file.read()
                    file.close()
                    words = [nltk.tag.str2tuple(t)[0] for t in data.split()]
                    sentence = ' '.join(words)
                    corpus.append(sentence)
                    Y.append(d)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray(),np.array(Y)
nested_cv()


# In[ ]:





# In[ ]:




