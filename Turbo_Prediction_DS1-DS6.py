#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:16:03 2019

@author: amar
"""
from Turbo_Prediction_NCI import NCI

import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
#import statistics as st
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(3)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel="linear", C=0.1,gamma=0.1,tol=0.01, probability=True)))
models.append(('LSVM', LinearSVC(C=0.1,tol=0.01, max_iter=1500)))
models.append(('RFC',RandomForestClassifier(n_estimators=100, random_state=7)))
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+--+-+-+-+-
def Tanimoto(data, Query, ActMap, Knn):
    """
:param data:(2D array) array contains fingerprint of molecules
:param Query:(1D array) array contains fingerprint of molecules as queries.
:param ActMap:(1D array) containes number repesent activity class type of each molecules in data param
:param Knn:(int) Is the requeted no of returen molecues at the top of similarity list as KNN                                      
                                    
:return lsim: list of Knn most similar molecules to query 
        cltyKnn: activity class for each returned Knn molecule in lsim
        MolPredict: activity class type for the molecule query
    """
    import numpy as np
    import pandas as pd    
      
    mol = data
    query = Query
    np.seterr(divide='ignore', invalid='ignore')
    #mol=np.nan_to_num(np.divide(data,data))
    # 
    mol=np.sqrt(mol)    
    #query=np.nan_to_num(np.divide(query,query))
    #
    query=np.sqrt(query) 
    
    m = mol.shape[0]
    q = query
    sim_score=np.zeros((m,1),dtype='float')
    for i in range(m):
        c = np.sum(mol[i,:]*q)
        a = np.sum(mol[i,:]**2)
        b = np.sum(q**2)
        sim_score[i] = float(c)/(a+b-c)
    return Ranking_score(data, sim_score, ActMap, Knn)
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+--+-+-+-+-
def Ranking_score(data, score, ActMap, Knn):
    
    """
    :param data:(1D array) array contains fingerprint of molecules
    :param score:(2D array) array contains similarity score of query to each molecule in data param                    
    :param ActMap:(1D array) containes number repesent activity class type of each molecules in data param
    :param Knn:(int) Is the requeted no of returen molecues at the top of similarity list as KNN                                      
                 
    :return lsim: list of Knn most similar molecules to query 
        cltyKnn: activity class for each returned Knn molecule in lsim
        MolPredict: activity class type for the molecule query
    """
    import numpy as np
    import pandas as pd  
    from collections import Counter       
    
    scoreidx = np.argsort(score,axis=0)     # sort list and return indices of desending sort list
    scoreidx = np.flipud(scoreidx)   # return fliup of list (reverse)
    lfing = data.shape[1]
    lsim1 = np.zeros((Knn,lfing),dtype='float')
    lsim2 = np.zeros((Knn,lfing),dtype='float')
    cltyKnn1 = list()
    cltyKnn2 = list()
    lpos1 = list()
    lpos2 = list()
    for i in range(Knn):
        indx1 = scoreidx[i]   # pickup the similar molecule from database            
        lpos1.append(indx1[0])  # save indx of each knn in the orginal database            
        lsim1[i,:]=data[indx1[0],:]           
        cltyKnn1.append(ActMap[indx1[0]])     # pickup the class type of similar molecule
            
        indx2 = scoreidx[-(Knn-i)]   # pickup the similar molecule from database
        lpos2.append(indx2[0])  # save indx of each knn in the orginal database
        lsim2[i,:]=data[indx2[0],:]
        cltyKnn2.append(ActMap[indx2[0]])     # pickup the class type of similar molecule
        
    clinact = Counter(cltyKnn2)  # class chosen of inact knn is the class of the last mol in sortig similarity score
    cldic = Counter(cltyKnn1)  # return  no of occurances of each class in dictionary 
    MolPredict = max(cldic, key=cldic.get) # return class of query based on class of max no of knn similar to him  
    clinact = max(clinact, key=clinact.get) # return class of query based on class of max no of knn similar to him
    return (lsim1, lpos1, cltyKnn1, cltyKnn1[0],lsim2, lpos2, clinact)# the knn similar molecules take class of first mol sim list (1knn) cltyKnn[0]    
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+--+-+-+-+-
def prediction_kvold(X_train, Y_train, cln):
    acc = []
    report = np.zeros((cln,4),dtype='float')
    mean_report = np.zeros(3,dtype='float')
    confusion_matrixs = np.zeros((cln,cln),dtype='int')
    best_svr = models[2][1] #LinearSVC(C=0.1,tol=0.01,max_iter=1500)#RandomForestClassifier(n_estimators=100, random_state=1)
    cv = model_selection.StratifiedKFold(n_splits=10)   

    for train_index, test_index in cv.split(X_train,Y_train):
        x_train, x_test, y_train, y_test = X_train[train_index], X_train[test_index], Y_train[train_index], Y_train[test_index]
        best_svr.fit(x_train, y_train)       
        predict = best_svr.predict(x_test)
        confusion_matrixs += confusion_matrix(y_test, predict)
        acc.append(balanced_accuracy_score(y_test, predict))
    
    TP = np.diag(confusion_matrixs)
    FP = np.sum(confusion_matrixs, axis=0) - TP
    FN = np.sum(confusion_matrixs, axis=1) - TP
    TN = []
    for i in range(cln):
        temp = np.delete(confusion_matrixs, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    accuracy = np.mean(acc)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    AUC = (specificity + recall)/2
    F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    
    report[:,0] = recall
    report[:,1] = specificity
    report[:,2] = F1
    report[:,3] = AUC
    mean_report[0] = np.mean(F1)
    mean_report[1] = np.mean(AUC)
    mean_report[2] = accuracy    
    return (report, mean_report)
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-    
def prediction_feedback_kfold(X, y, knn, cln):
    acc = []
    report = np.zeros((cln,4),dtype='float')
    mean_report = np.zeros(3,dtype='float')
    confusion_matrixs = np.zeros((cln,cln),dtype='int')
    best_svr = models[2][1] #LinearSVC(C=0.1,tol=0.01,max_iter=1500)#RandomForestClassifier(n_estimators=100, random_state=1)
    cv = model_selection.StratifiedKFold(n_splits=10)
    for train_index, test_index in cv.split(X,y):
        X_train, X_validation, Y_train, Y_validation = X[train_index], X[test_index], y[train_index], y[test_index]        
        predict =np.zeros(len(Y_validation),dtype='int')
        indx = 0
        for q in X_validation:
            result = Tanimoto(X_train, q, Y_train, knn)
            fdbk_pst = result[1] # add knn molecules return by Tanimoto for each query q to list 
            fdbk_Predict = result[3] # add class type return by Tanimoto as class for knn of each query q to list
            Y_train1 = np.copy(Y_train)
            Y_train1[fdbk_pst] = fdbk_Predict # change class of feedback knn in the training database
            best_svr.fit(X_train, Y_train1)
            predictions = best_svr.predict(q.reshape(1,q.shape[0]))          
            predict[indx] = predictions[0]
            indx +=1          
        confusion_matrixs += confusion_matrix(Y_validation, predict)
        acc.append(balanced_accuracy_score(Y_validation, predict))
        
    TP = np.diag(confusion_matrixs)
    FP = np.sum(confusion_matrixs, axis=0) - TP
    FN = np.sum(confusion_matrixs, axis=1) - TP
    TN = []
    for i in range(cln):
        temp = np.delete(confusion_matrixs, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    accuracy = np.mean(acc)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    AUC = (specificity + recall)/2
    F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    
    report[:,0] = recall
    report[:,1] = specificity
    report[:,2] = F1
    report[:,3] = AUC
    mean_report[0] = np.mean(F1)
    mean_report[1] = np.mean(AUC)
    mean_report[2] = accuracy
    return (report, mean_report)
#++++++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
def prediction_reforming_query_kfold(X, y, knn, cln):
    acc = []
    report = np.zeros((cln,4),dtype='float')
    mean_report = np.zeros(3,dtype='float')
    confusion_matrixs = np.zeros((cln,cln),dtype='int')
    best_svr = models[2][1] #LinearSVC(C=0.1,tol=0.01,max_iter=1500)#RandomForestClassifier(n_estimators=100, random_state=1)
    cv = model_selection.StratifiedKFold(n_splits=10)    
    for train_index, test_index in cv.split(X,y):
        X_train, X_validation, Y_train, Y_validation = X[train_index], X[test_index], y[train_index], y[test_index]        
        best_svr.fit(X_train, Y_train)
        X_validation_reforming = np.zeros(X_validation.shape,dtype='float')
        for i in range(len(X_validation)):
            result = Tanimoto(X_train, X_validation[i,:], Y_train, knn)
            New_q = np.row_stack((X_validation[i,:],result[0])) # add query q to the array of knn molecules (return from Tanimoto function) 
            bits_on = np.count_nonzero(result[0], axis=0)
            bits_average_New_q = np.average(New_q, axis=0)
            bits_average_knn = np.average(result[0], axis=0)
            for j in range(len(X_validation[i,:])):
                if X_validation[i,j]>0 and bits_on[j]>0:
                    X_validation_reforming[i,j] = bits_average_New_q[j]
                elif X_validation[i,j]==0 and bits_on[j]>=(70*knn/100):
                    X_validation_reforming[i,j] = bits_average_knn[j]
        
        predict = best_svr.predict(X_validation_reforming)
        confusion_matrixs += confusion_matrix(Y_validation, predict)
        acc.append(balanced_accuracy_score(Y_validation, predict))
   
    TP = np.diag(confusion_matrixs)
    FP = np.sum(confusion_matrixs, axis=0) - TP
    FN = np.sum(confusion_matrixs, axis=1) - TP
    TN = []
    for i in range(cln):
        temp = np.delete(confusion_matrixs, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    accuracy = np.mean(acc)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    AUC = (specificity + recall)/2
    F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    
    report[:,0] = recall
    report[:,1] = specificity
    report[:,2] = F1
    report[:,3] = AUC
    mean_report[0] = np.mean(F1)
    mean_report[1] = np.mean(AUC)
    mean_report[2] = accuracy
    return (report, mean_report)                
#+++++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-                             
def prediction_feedback_multiquery_kfold(X, y, knn, cln):
    acc = []
    report = np.zeros((cln,4),dtype='float')
    mean_report = np.zeros(3,dtype='float')
    confusion_matrixs = np.zeros((cln,cln),dtype='int')
    best_svr = models[2][1] #LinearSVC(C=0.1,tol=0.01,max_iter=1500)#RandomForestClassifier(n_estimators=100, random_state=1)
    cv = model_selection.StratifiedKFold(n_splits=10)
    for train_index, test_index in cv.split(X,y):
        X_train, X_validation, Y_train, Y_validation = X[train_index], X[test_index], y[train_index], y[test_index]                
        indx =0
        predict =np.zeros(len(Y_validation),dtype='int')
        confusion_matrx = np.zeros((cln, cln), dtype='int')
        best_svr.fit(X_train, Y_train)
        for i in range(len(X_validation)):
            result = Tanimoto(X_train, X_validation[i,:], Y_train, knn)
            X_validation_feedback = np.row_stack((X_validation[i,:],result[0])) # add query q to the array of knn molecules              
            predictions = best_svr.predict(X_validation_feedback)
            classes = Counter(predictions)
            pr_class = max(classes, key = classes.get)
            confusion_matrx[Y_validation[i]-1,pr_class-1]+=1
            predict[indx] = pr_class
            indx +=1            
            
        confusion_matrixs += confusion_matrx
        acc.append(balanced_accuracy_score(Y_validation, predict))
            
    TP = np.diag(confusion_matrixs)
    FP = np.sum(confusion_matrixs, axis=0) - TP
    FN = np.sum(confusion_matrixs, axis=1) - TP
    TN = []
    for i in range(cln):
        temp = np.delete(confusion_matrixs, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    accuracy = np.mean(acc)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    AUC = (specificity + recall)/2
    F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    
    report[:,0] = recall
    report[:,1] = specificity
    report[:,2] = F1
    report[:,3] = AUC
    mean_report[0] = np.mean(F1)
    mean_report[1] = np.mean(AUC)
    mean_report[2] = accuracy
  
    return (report, mean_report)      
#++++++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-                    
def prediction(X_train, Y_train, X_validation, Y_validation, cln):
    report = np.zeros((cln,4),dtype='float')
    mean_report = np.zeros(3,dtype='float')
    best_svr = models[2][1] #LinearSVC(C=0.1,tol=0.01,max_iter=1500)#RandomForestClassifier(n_estimators=100, random_state=1)
    best_svr.fit(X_train, Y_train)
    predict = best_svr.predict(X_validation)
    accuracy = balanced_accuracy_score(Y_validation, predict)
    confusion_matrixs = confusion_matrix(Y_validation, predict)
    
    TP = np.diag(confusion_matrixs)
    FP = np.sum(confusion_matrixs, axis=0) - TP
    FN = np.sum(confusion_matrixs, axis=1) - TP
    TN = []
    for i in range(cln):
        temp = np.delete(confusion_matrixs, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    AUC = (specificity + recall)/2
    
    F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    report[:,0] = recall
    report[:,1] = specificity
    report[:,2] = F1
    report[:,3] = AUC
    mean_report[0] = np.mean(F1)
    mean_report[1] = np.mean(AUC)
    mean_report[2] = accuracy
    return (report, mean_report)
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+-+-+-+    
def prediction_feedback(X_train, Y_train, X_validation, Y_validation, knn, cln):#data, queries, ActMap_data, ActMap_queries, Knn):
    """
    prediction_feedback_v1: this version used copy of unsorted data as original       
    """
    report = np.zeros((cln,4),dtype='float')
    mean_report = np.zeros(3,dtype='float')
    predict =np.zeros(len(Y_validation),dtype='int')    
    indx = 0    
    for q in X_validation:
        result = Tanimoto(X_train, q, Y_train, knn)
        fdbk_pst = result[1] # add knn molecules return by Tanimoto for each query q to list 
        fdbk_Predict=result[3] # add class type return by Tanimoto as class for knn of each query q to list
        Y_train1 = np.copy(Y_train)
        Y_train1[fdbk_pst] = fdbk_Predict # change class of feedback knn in the training database
        best_svr = models[2][1] #LinearSVC(C=0.1,tol=0.01,max_iter=1500)#RandomForestClassifier(n_estimators=100, random_state=1)
        best_svr.fit(X_train, Y_train1)            
        predictions = best_svr.predict(q.reshape(1,q.shape[0]))          
        predict[indx] = predictions[0]
        indx +=1
        
    accuracy = balanced_accuracy_score(Y_validation, predict)
    confusion_matrixs = confusion_matrix(Y_validation, predict)
  
    TP = np.diag(confusion_matrixs)
    FP = np.sum(confusion_matrixs, axis=0) - TP
    FN = np.sum(confusion_matrixs, axis=1) - TP
    TN = []
    for i in range(cln):
        temp = np.delete(confusion_matrixs, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    AUC = (specificity + recall)/2
    F1 = np.nan_to_num(np.array(2 * (precision * recall) / (precision + recall)))
    
    report[:,0] = recall
    report[:,1] = specificity
    report[:,2] = F1
    report[:,3] = AUC
    mean_report[0] = np.mean(F1)
    mean_report[1] = np.mean(AUC)
    mean_report[2] = accuracy
    return (report, mean_report)
#+++++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-       
def prediction_reforming_query(X_train, Y_train, X_validation, Y_validation, knn, cln):#data, queries, ActMap_data, ActMap_queries, Knn):    
    report = np.zeros((cln,4),dtype='float')
    mean_report = np.zeros(3,dtype='float')
    X_validation_reforming = np.zeros(X_validation.shape,dtype='float')
       
    for i in range(len(X_validation)):
        result = Tanimoto(X_train, X_validation[i,:], Y_train, knn)
        knn_q = np.row_stack((X_validation[i,:],result[0])) # add query q to the array of knn molecules 
        bits_on = np.count_nonzero(result[0], axis=0)
        bits_average_knn_q = np.average(knn_q, axis=0)
        bits_average_knn = np.average(result[0], axis=0)
        for j in range(len(X_validation[i,:])):
            if X_validation[i,j]>0 and bits_on[j]>0:
                X_validation_reforming[i,j] = bits_average_knn_q[j]
            elif X_validation[i,j]==0 and bits_on[j]>=(70*knn/100):
                X_validation_reforming[i,j] = bits_average_knn[j]

    best_svr = models[2][1] #LinearSVC(C=0.1,tol=0.01,max_iter=1500)#RandomForestClassifier(n_estimators=100, random_state=1)
    best_svr.fit(X_train, Y_train)
    predict = best_svr.predict(X_validation_reforming)
    
    accuracy = balanced_accuracy_score(Y_validation, predict)
    confusion_matrixs = confusion_matrix(Y_validation, predict)
    
    TP = np.diag(confusion_matrixs)
    FP = np.sum(confusion_matrixs, axis=0) - TP
    FN = np.sum(confusion_matrixs, axis=1) - TP
    TN = []
    for i in range(cln):
        temp = np.delete(confusion_matrixs, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    AUC = (specificity + recall)/2
    F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    
    report[:,0] = recall
    report[:,1] = specificity
    report[:,2] = F1
    report[:,3] = AUC
    mean_report[0] = np.mean(F1)
    mean_report[1] = np.mean(AUC)
    mean_report[2] = accuracy
    return (report, mean_report)

#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def prediction_feedback_multiquery(X_train, Y_train, X_validation, Y_validation, knn, cln):#data, queries, ActMap_data, ActMap_queries, Knn):
    report = np.zeros((cln,4),dtype='float')
    mean_report = np.zeros(3,dtype='float')
    confusion_matrixs = np.zeros((cln, cln), dtype='int')
    predict =np.zeros(len(Y_validation),dtype='int')
    
    for i in range(len(X_validation)):
        result = Tanimoto(X_train, X_validation[i,:], Y_train, knn)
        X_validation_feedback = np.row_stack((X_validation[i,:],result[0])) # add query q to the array of knn molecules 
        
        best_svr = models[2][1] #LinearSVC(C=0.1,tol=0.01,max_iter=1500)#RandomForestClassifier(n_estimators=100, random_state=1)
        best_svr.fit(X_train, Y_train)
        predictions = best_svr.predict(X_validation_feedback)
        classes = Counter(predictions)
        pr_class = max(classes, key = classes.get)
        confusion_matrixs[Y_validation[i]-1,pr_class-1]+=1
        predict[i] = pr_class 
     
    accuracy = balanced_accuracy_score(Y_validation, predict)
    
    TP = np.diag(confusion_matrixs)
    FP = np.sum(confusion_matrixs, axis=0) - TP
    FN = np.sum(confusion_matrixs, axis=1) - TP
    TN = []
    for i in range(cln):
        temp = np.delete(confusion_matrixs, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    AUC = (specificity + recall)/2
    F1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
    report[:,0] = recall
    report[:,1] = specificity
    report[:,2] = F1
    report[:,3] = AUC
    mean_report[0] = np.mean(F1)
    mean_report[1] = np.mean(AUC)
    mean_report[2] = accuracy
    
    return (report, mean_report)
#+++++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-       
def DS6():
     # Building and evaluating classification algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier(3)))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(kernel="linear", C=0.1,gamma=0.1,tol=0.01, probability=True)))
    models.append(('LSVM', LinearSVC(C=0.1,tol=0.01, max_iter=1500)))
    models.append(('RFC',RandomForestClassifier(n_estimators=100, random_state=7)))
         
    os.chdir('/home/amar/Turbo/NCI_balanced')
    fno =1024 # length of fingerprint
    mno = [3586, 2934, 2700, 3470, 4162, 3918, 3546, 5430, 3282]  # no of molecules in database
    cln = 2 # no of classes
    neighbour =[5, 10]
    for i in range(1,10):
        data = str(i) + '_ECFC4_'+str(mno[i-1])
        dataset = pd.read_csv(data+'x'+str(fno+1)+'.txt', sep='\t', header = None)
        dataset.head()

        # split dataset into train, test and validation sets
        array = dataset.values
        X =array[:,0:fno]
        y =array[:,fno]
        
        np.seterr(divide='ignore', invalid='ignore')
        X = np.nan_to_num(np.divide(X,X))
        
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,y, test_size=0.2, random_state=1)    
        
        data = data +'_'  
        name = '_KNeighbors'
        for knn in neighbour:#[1:]:
            
            with open(data+str(knn)+ name  + '.txt','w') as output:
                output.write(str(prediction_kvold(X_train, Y_train, cln)[1]))
                output.write("\n")
                output.write(str(prediction_feedback_kfold(X_train, Y_train, knn, cln)[1]))
                output.write("\n")
                output.write(str(prediction_reforming_query_kfold(X_train, Y_train, knn, cln)[1]))
                output.write("\n")
                output.write(str(prediction_feedback_multiquery_kfold(X_train, Y_train, knn, cln)[1]))
        
                output.write("\n")
                output.write("\n")
        for knn in neighbour:#[0:-1]:
            with open(data+str(knn)+ name  + '.txt','a') as output:
                x1 = prediction(X_train, Y_train, X_validation, Y_validation, cln)
                
                output.write(str(x1[1]))
                output.write("\n")  
                x2 = prediction_feedback(X_train, Y_train, X_validation, Y_validation, knn, cln)
                output.write(str(x2[1]))
                output.write("\n")
                x3 = prediction_reforming_query(X_train, Y_train, X_validation, Y_validation, knn, cln)
                output.write(str(x3[1]))
                output.write("\n")
                x4 = prediction_feedback_multiquery(X_train, Y_train, X_validation, Y_validation, knn, cln)
                output.write(str(x4[1]))
                output.write("\n")
        
            with open(data+str(knn)+ name  + '_activity.txt','w') as outp:
                outp.write(str(x1[0]))
                outp.write("\n")  
                outp.write(str(x2[0]))
                outp.write("\n")
                outp.write(str(x3[0]))
                outp.write("\n")
                outp.write(str(x4[0]))
                outp.write("\n") 
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+--+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def DS1_DS5():
    # Building and evaluating classification algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier(3)))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(kernel="linear", C=0.1,gamma=0.1,tol=0.01, probability=True)))
    models.append(('LSVM', LinearSVC(C=0.1,tol=0.01, max_iter=1500)))
    models.append(('RFC',RandomForestClassifier(n_estimators=100, random_state=7)))
    
    #os.chdir('/home/amar/Turbo/MDDR/DS1')
    #os.chdir('/home/amar/Turbo/DUD')
    os.chdir('/home/amar/Turbo/MUV')
    fno =1024 # length of fingerprint
    mno = 510  # no of molecules in database
    cln = 17 # no of classes

    data = 'MUV_ECFC4_'
    dataset = pd.read_csv(data+str(mno)+'x'+str(fno+1)+'.txt', sep='\t', header = None)
    dataset.head()
    
    # split dataset into train, test and validation sets
    array = dataset.values
    X =array[:,0:fno]
    y =array[:,fno]
    
    np.seterr(divide='ignore', invalid='ignore')
    X = np.nan_to_num(np.divide(X,X))  # convert to from integer fingerprint to binary fingerprint
        
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,y, test_size=0.2, random_state=1)
        
    neighbour =[5, 10]
    data = data + str(mno) +'_'
    name = '_KNeighbors'

    for knn in neighbour:
        
        with open(data+str(knn)+ name  + '.txt','w') as output:
            output.write(str(prediction_kvold(X_train, Y_train, cln)[1]))
            output.write("\n")
            output.write(str(prediction_feedback_kfold(X_train, Y_train, knn, cln)[1]))
            output.write("\n")
            output.write(str(prediction_reforming_query_kfold(X_train, Y_train, knn, cln)[1]))
            output.write("\n")
            output.write(str(prediction_feedback_multiquery_kfold(X_train, Y_train, knn, cln)[1]))
        
            output.write("\n")
            output.write("\n")
   
    for knn in neighbour:
        with open(data+str(knn)+ name  + '.txt','a') as output:
            x1 = prediction(X_train, Y_train, X_validation, Y_validation, cln)
        
            output.write(str(x1[1]))
            output.write("\n")  
            x2 = prediction_feedback(X_train, Y_train, X_validation, Y_validation, knn, cln)
            output.write(str(x2[1]))
            output.write("\n")
            x3 = prediction_reforming_query(X_train, Y_train, X_validation, Y_validation, knn, cln)
            output.write(str(x3[1]))
            output.write("\n")
            x4 = prediction_feedback_multiquery(X_train, Y_train, X_validation, Y_validation, knn, cln)
            output.write(str(x4[1]))
            output.write("\n")
        
            with open(data+str(knn)+ name  + '_activity.txt','w') as outp:
                outp.write(str(x1[0]))
                outp.write("\n")  
                outp.write(str(x2[0]))
                outp.write("\n")
                outp.write(str(x3[0]))
                outp.write("\n")
                outp.write(str(x4[0]))
                outp.write("\n")
