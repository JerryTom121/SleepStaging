"""
Data Preparation. Based on the designed experiment, the script stacks features and labels, and then writes them 
into .csv files which are then used for learning and validation;.csv files get their names based on the experiment number.

@param num_exp => the number of experiment

"""
import numpy as np
import sys 
sys.path.insert(0, '../library')
from sklearn.cross_validation import train_test_split
from Experiment import Experiment 
from features import extract_features
from artlib import remove_artefakts 
from sklearn.externals import joblib
# --------------------------------------------------------------------------- #
# -- Read command line arguments -------------------------------------------- #
# --------------------------------------------------------------------------- #
param1 = sys.argv[1]
# --------------------------------------------------------------------------- #
# -- Choose mapping based on the classification goal: ----------------------- #
# -- do we want to detect anomalies or classify sleeping stages ------------- #
# --------------------------------------------------------------------------- #
MAPPING = { "w": +1, "n": +1, "r": +1,                               # normal
            "1": -1, "2": -1, "3": -1, "a": -1, "'": -1, "4": -1 }   # artefact       
# --------------------------------------------------------------------------- #
# --------------------- Load Experiment ------------------------------------- #
# --------------------------------------------------------------------------- #
Exp = Experiment(param1)
print "## Performing data generation for experiment number:  " + str(param1)
# --------------------------------------------------------------------------- #
# ----------------- Extract features and labels ----------------------------- #
# --------------------------------------------------------------------------- #
# Initial scaler is nul
scaler_train = None
print "## Reading training and validation data..."
if Exp.split:
    # Get both training and validation data
    [features,labels] = extract_features(Exp,MAPPING,Exp.trainset)
    # Remove artefakts if requested by the experiment
    if Exp.artrem:
        print '## Removing artifacts from data as specified...'
        [features,labels] = remove_artefakts(features,labels)
    # Split the data into training and validation
    if Exp.rndsplit==False:                 # regular split 75%-25%
        tra_size = len(labels)-len(labels)/4
        Xt = features[0:tra_size]
        Xv = features[tra_size:len(labels)]
        Yt = labels[0:tra_size]
        Yv = labels[tra_size:len(labels)]
    else:                                   # random split 75%-25%
        Xt, Xv, Yt, Yv = train_test_split(features, labels, test_size=0.25, random_state=42)

else:
    # Get training data
    [Xt,Yt,scaler_train] = extract_features(Exp,MAPPING,Exp.trainset)
    # Save or load scaler
    if scaler_train==None:
        print "## Load scaler -- no training data...."
        joblib.load('scaler_'+param1+'.pkl') 
    else:
        print "## Save scaler -- learned from training data...."
        joblib.dump(scaler_train, 'scaler_'+param1+'.pkl')
    # Get validation data
    [Xv,Yv,scaler_test]  = extract_features(Exp,MAPPING,Exp.testset,scaler_train)
    # Remove artefakts if requested by the experiment
    if Exp.artrem:
        print '## Removing artifacts from data as specified...'
        [Xt,Yt] = remove_artefakts(Xt,Yt)
        [Xv,Yv] = remove_artefakts(Xv,Yv)
# PAY ATTENTION !!! Allow only one label column fo training data; We can have multiple columns
# in validation data for derriving different statistics
if np.shape(Yt)[0]>0:
    Yt = Yt[:,0]
# Adding the id-column to the data
idx_Xt = np.linspace(1, len(Xt), len(Xt)).astype(np.int);
idx_Xv = np.linspace(1 + len(Xt), len(Xt) + len(Xv), len(Xv)).astype(np.int);
#Xt = np.c_[idx_Xt, Xt];
#Xv = np.c_[idx_Xv, Xv];
# --------------------------------------------------------------------------- #
# ----------------- Save prepared data -------------------------------------- #
# --------------------------------------------------------------------------- #
training_data = np.c_[idx_Xt, Xt, Yt]
print "## Saving training data... (" + str(np.shape(training_data)) + ")"
np.savetxt(Exp.csv_folder+'train_exp'+str(param1)+'.csv', training_data, fmt='%s', delimiter=',')
#
validation_data = np.c_[idx_Xv, Xv, Yv]
print "## Saving validation data... (" + str(np.shape(validation_data)) + ")"
np.savetxt(Exp.csv_folder+'test_exp' +str(param1)+'.csv', validation_data, fmt='%s', delimiter=',')
