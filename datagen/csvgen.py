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


# --------------------------------------------------------------------------- #
# -- Choose mapping based on the classification goal: ----------------------- #
# -- do we want to detect anomalies or classify sleeping stages ------------- #
# --------------------------------------------------------------------------- #
MAPPING = { "w": +1, "n": +1, "r": +1,                               # normal
            "1": -1, "2": -1, "3": -1, "a": -1, "'": -1, "4": -1 }   # artifact
# --------------------------------------------------------------------------- #
# --------------------- Load Experiment ------------------------------------- #
# --------------------------------------------------------------------------- #
Exp = Experiment(sys.argv[1])
print "-------------------------------"
print "The chosen experiment is " + str(sys.argv[1])
print "-------------------------------"
# --------------------------------------------------------------------------- #
# ----------------- Extract features and labels ----------------------------- #
# --------------------------------------------------------------------------- #
if Exp.split:
    '''
    In case we do not have separate testing data set
    we allocate a part of training data for testing purposes
    '''
    print 'Reading all data set from the same file set:'
    print '--------------------------------------------'
    print 'Get data set..'
    [features,labels] = extract_features(Exp.feat_ext,Exp.eeg_folder,Exp.trainset,MAPPING,Exp.interval,Exp.extrain)
    # Remove artefakts if requested by the experiment
    if Exp.artrem:
        print 'Remove artefakts from training data set!'
        [features,labels] = remove_artefakts(features,labels)

    print 'Split data set into testing and training..'
    if Exp.rndsplit==False:
        print "Use non-random split.."
        tra_size = len(labels)-len(labels)/4
        Xt = features[0:tra_size]
        Xv = features[tra_size:len(labels)]
        Yt = labels[0:tra_size]
        Yv = labels[tra_size:len(labels)]
    else:
        print "Use random split.."
        Xt, Xv, Yt, Yv = train_test_split(features, labels, test_size=0.25, random_state=42)

else:
    print 'Reading training and testing data set from the separate file sets:'
    print '------------------------------------------------------------------'
    print 'Get training data set..'
    [Xt,Yt] = extract_features(Exp.feat_ext,Exp.eeg_folder,Exp.trainset,MAPPING,Exp.interval,Exp.extrain)
    print 'Get testing data set..'
    [Xv,Yv] = extract_features(Exp.feat_ext,Exp.eeg_folder,Exp.testset,MAPPING,Exp.interval,Exp.extest)
    # Remove artefakts if requested by the experiment
    if Exp.artrem:
        print 'Remove artefakts from both data sets!'
        [Xt,Yt] = remove_artefakts(Xt,Yt)
        [Xv,Yv] = remove_artefakts(Xv,Yv)


# PAY ATTENTION !!! Allow only one column fo training data
Yt = Yt[:,0]

# Adding the id-column to the data
idx_Xt = np.linspace(1, len(Xt), len(Xt)).astype(np.int);
idx_Xv = np.linspace(1 + len(Xt), len(Xt) + len(Xv), len(Xv)).astype(np.int);
Xt = np.c_[idx_Xt, Xt];
Xv = np.c_[idx_Xv, Xv];

# Save data
np.savetxt(Exp.csv_folder+'train_exp'+str(sys.argv[1])+'.csv', np.c_[Xt, Yt], fmt='%s', delimiter=',')
np.savetxt(Exp.csv_folder+'test_exp' +str(sys.argv[1])+'.csv', np.c_[Xv, Yv], fmt='%s', delimiter=',')
