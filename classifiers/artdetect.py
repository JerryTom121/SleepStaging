"""
Script used for testing the artefakt identification.
Calls functions from 'artlib' which try to detect artefakts.
"""
import sys; sys.path.insert(0, '../library')
from edfplus import load_edf 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# sklearn stuff
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC,OneClassSVM
from sklearn import preprocessing
from sklearn.decomposition import PCA,FastICA
# hmm library
from hmmlearn import hmm
# signal processing stuff
from scipy import signal
# my stuff
import artlib
import sleeplib
import features

# Some formatting configuration..
# ################################
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


# --------------------------------------------------------------------------- #
# ---------------------- Experimental setup --------------------------------- #
# --------------------------------------------------------------------------- #
NEXP = '5'
aug  = '_aug'
CSV_FOLDER = '/home/djordje/Desktop/CSVdata/'
# --------------------------------------------------------------------------- #
# ---------------------- Read data ------------------------------------------ #
# --------------------------------------------------------------------------- #
train_mat = np.genfromtxt(CSV_FOLDER+'train_exp'+NEXP+aug+'.csv', skip_header=0, dtype=str, comments="4s",delimiter=',')
test_mat  = np.genfromtxt(CSV_FOLDER+'test_exp'+NEXP+'.csv', skip_header=0, dtype=str, comments="4s",delimiter=',')
# Training set
train_data  = (train_mat[:,1:-1]).astype(float)
train_label = (train_mat[:,-1]).astype(float)
# Testing set
test_data  = (test_mat[:,1:-3]).astype(float)
test_label = (test_mat[:,-3]).astype(float)
# DEBUG
print "Data loaded.."
# --------------------------------------------------------------------------- #
# ---------------------- Normalization -------------------------------------- #
# --------------------------------------------------------------------------- #
scaler = preprocessing.RobustScaler()
train_data = scaler.fit_transform(train_data)
test_data  = scaler.transform(test_data)
print "Normalization performed.."
# --------------------------------------------------------------------------- #
# ---------------------- Choose classifier ---------------------------------- #
# --------------------------------------------------------------------------- #
# Random forest
#clf  = RandomForestClassifier()
#clf  = GradientBoostingClassifier(n_estimators=1000)
#clf  = LogisticRegression()
clf = SVC(C=5)
# --------------------------------------------------------------------------- #
# ---------------------- Test classification -------------------------------- #
# --------------------------------------------------------------------------- #
# Fit
clf.fit(train_data,train_label)
# Predict
predictions = clf.predict(test_data)
# Evaluate quality of predictions
hits = 0
miss = 0
fpos = 0
for i in range(len(predictions)):

    # Artifact
    if test_label[i] == -1:
        # Detected
        if predictions[i] == -1:
            hits = hits + 1
        # Missed
        else:
            miss = miss + 1
    # Non-artifact
    else:
        # False positive
        if predictions[i] == -1:
            fpos = fpos + 1

# print stats
print("Hits: "+str(hits))
print("Fpos: "+str(fpos))
print("Miss: "+str(miss))