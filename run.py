# Author: Djordje Miladinovic
# License:

import config as cfg
import os
import sys
import subprocess
import numpy as np
from sklearn.externals import joblib
from sklearn import metrics
import sslib.preprocessing as prep
import sslib.parsing as pars
from sslib.shallow import train as shtrain

# --------------------------------------------------------------------------- #
# -- UZH format specific initialization ------------------------------------- #
# --------------------------------------------------------------------------- #
# TODO: make configurable formatting
FeatureExtractor = prep.FeatureExtractorUZH
ScoringParser = pars.ScoringParserUZH
architecture = "temporal_convolution_UZH_"+cfg.PROBLEM
signal_length = 512
num_channels = 3

# ---------------------------------------------------------------------------- #
# ----- Utility functions ---------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def add_neighbors(features, num_neighbors):
    # TODO: implement this more elegantly
    """Extend feature matrix to surround each sample by num_neighbors/2 on front
    and num_neighbors/2 on the back.
    """
    # number of neighbors must be even (bi-directional context)
    assert(number_neighbors % 2 == 0)
    # initialize feature matrix
    M = np.array([])
    for c in range(num_channels):
        A = features[:, c*signal_length:(c+1)*signal_length]
        T = A.copy()
        for n in range(num_neighbors/2):
            prev = np.roll(A, shift=(n+1), axis=0)
            subq = np.roll(A, shift=-(n+1), axis=0)
            T = np.hstack((prev, T, subq))
        M = np.hstack((M, T)) if M.size else T
    # return newly constructed feature matrix
    return M

def generate_csv(datapath, scaler=None):
    # TODO: make it more elegant - avoid 'if'
    # TODO: leaving out ambigious samples needs to be cleaner
    """Given the path to data, parse raw recordings and scorings, merge these
    and write into a .csv file.
    """
    # Acquire recordings
    recordings = []
    for recording in os.listdir(datapath['recordings']):
        recordings.append(datapath['recordings']+recording)
    t_features = FeatureExtractor(recordings).get_temporal_features() # 3 channels
    #f_features = FeatureExtractor(recordings).get_fourier_features() # 3 channels (so far)
    #features = np.hstack((t_features,f_features))
    features = t_features
    # Sanity check
    assert(np.shape(features)[1]==signal_length*num_channels)
    # Acquire scorings
    scorings = []
    for scoring in os.listdir(datapath['scorings']):
        scorings.append(datapath['scorings']+scoring)
    if cfg.PROBLEM == "AD":
        labels = ScoringParser(scorings).get_binary_scorings()
    elif cfg.PROBLEM == "SS":
        labels = ScoringParser(scorings).get_4stage_scorings()
    # Sanity check: #labels is equal to #features
    assert(np.shape(features)[0]==len(labels))
    # Leave out ambigious samples
    features = features[labels.flatten()<4,:]
    labels = labels[labels.flatten()<4]
    # Normalize if scaler is given, otherwise make it
    if scaler==None:
        scaler = prep.NormalizerZMUV(num_channels)
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    # Add neighbors to capture the context
    features = add_neighbors(features, cfg.num_neighbors)
    # Concatenate features and labels, and save the data
    dataset = np.hstack((features,labels))
    # Save constructed dataset into a .csv file
    np.savetxt(datapath['csv'], dataset, delimiter=",")
    print "# The shape of data is: " + str(np.shape(dataset))
    # Return scaler
    return scaler

# ---------------------------------------------------------------------------- #
# ----- Main functions ------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def prepare():
    """ Given raw recordings and scorings file generate 3 different data sets in
    .csv format: training set, holdout set and testing set.
    """    
    print "## Generate training data set:"
    scaler = generate_csv(cfg.TRAINSET)
    print "## Generate holdout data set:"
    generate_csv(cfg.HOLDOUT, scaler)
    print "## Generate test data set:"
    generate_csv(cfg.TESTSET, scaler)


def train(gpu):
    """Train model on specified GPU using previously prepared training and holdout
    data sets. Essentially we call corresponding .lua script in torch framework 
    and we use previously configured training parameters ('config.py' file).
    """
    os.system('CUDA_VISIBLE_DEVICES='+gpu+' '+'th sslib/deepnet/train.lua'\
              +' -learningRate '+str(cfg.learning_rate)\
              +' -learningRateDecay '+str(cfg.learning_rate_decay)\
              +' -momentum '+str(cfg.momentum)\
              +' -weightDecay '+str(cfg.weight_decay)\
              +' -dropout '+str(cfg.dropout)\
              +' -batchSize '+str(cfg.batch_size)\
              +' -maxEpochs '+str(cfg.max_epochs)\
              +' -nclasses '+str(cfg.num_classes)\
              +' -trainPath '+cfg.TRAINSET['csv']\
              +' -holdoutPath '+cfg.HOLDOUT['csv']\
              +' -architecture '+architecture\
              +' -gpu '+gpu\
              +' -inputSize '+str(signal_length*(1+cfg.num_neighbors))\
              +' -numChannels '+str(num_channels))

# ---------------------------------------------------------------------------- #
# - Parse command to process data, train a model or evaluate already trained - #
# ---------------------------------------------------------------------------- #
command = sys.argv[1]
gpu = sys.argv[2] if len(sys.argv)>2 else "0"
if command == 'prepare':
    prepare()
elif command == 'train':
    train(gpu)
else:
    print "Unknown Command!"
