"""
Script used for testing the artefakt identification.
Calls functions from 'artlib' which try to detect artefakts.

@param num_exp => the number of experiment
"""
import sys; sys.path.insert(0, '../library')
from edfplus import load_edf 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# sklearn stuff
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
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
NEXP = str(sys.argv[1])
aug  = '_aug'
CSV_FOLDER = '/home/djordje/Desktop/CSVdata/'
# --------------------------------------------------------------------------- #
# ---------------------- Read data ------------------------------------------ #
# --------------------------------------------------------------------------- #
train_mat = np.genfromtxt(CSV_FOLDER+'train_exp'+NEXP+aug+'.csv', skip_header=0, dtype=str, comments="4s",delimiter=',')
test_mat  = np.genfromtxt(CSV_FOLDER+'test_exp'+NEXP+'.csv', skip_header=0, dtype=str, comments="4s",delimiter=',')
# Training set
print "Load training data..."
train_data  = (train_mat[:,1:-1]).astype(float)
train_label = (train_mat[:,-1]).astype(float)
print "Shape is: "
print np.shape(train_data)
# Testing set
print "Load validation data..."
test_data  = (test_mat[:,1:-3]).astype(float)
test_label = (test_mat[:,-3]).astype(float)
print "Shape is: "
print np.shape(test_data)
print "----------------------------"
print "-- The augmentation type: " + aug 
print "----------------------------"
# --------------------------------------------------------------------------- #
# ---------------------- Choose classifier ---------------------------------- #
# --------------------------------------------------------------------------- #
# Random forest
#clf  = RandomForestClassifier();                  print "Random Forest..."
#clf  = GradientBoostingClassifier(n_estimators=1000)
clf  = LogisticRegression();                      print"Logistic Regression..."
#clf  = AdaBoostClassifier(n_estimators=200);       print"AdaBoost - 200"
#clf = SVC(C=5);										print"SVM..."
# --------------------------------------------------------------------------- #
# ---------------------- Test classification -------------------------------- #
# --------------------------------------------------------------------------- #
# Fit
clf.fit(train_data,train_label)
# Evaluate predictions
print "Evaluate on test set"
sleeplib.evaluate(clf.predict(test_data),test_label)
#print "Evaluate on train set"
#sleeplib.evaluate(clf.predict(train_data),train_label)

