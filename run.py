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
    for i in range(num_neighbors/2):
        features_prev = np.roll(features, shift=i, axis=0)
        features_subq = np.roll(features, shift=-i, axis=0)
        # stack
        features = np.hstack((features_prev[:, 0:signal_length], features[:, 0:signal_length], features_subq[:, 0:signal_length],\
                              features_prev[:,   signal_length:2*signal_length], features[:,   signal_length:2*signal_length], features_subq[:,   signal_length:2*signal_length],\
                              features_prev[:, 2*signal_length:3*signal_length], features[:, 2*signal_length:3*signal_length], features_subq[:, 2*signal_length:3*signal_length]))
    return features

def generate_csv(datapath, scaler=None): 
    # TODO: make it more elegant-avoid 'if'
    # TODO: enable simple configuration of feature selection method
    """Given the path to data, parse raw recordings and scorings, merge these
    and write into a .csv file.
    """
    # Parse features from raw data files
    recordings = []
    for recording in os.listdir(datapath['recordings']):
        recordings.append(datapath['recordings']+recording)
    features = FeatureExtractor(recordings).get_temporal_features()
    # Normalize if scaler is given, otherwise make it
    if scaler==None:
        scaler = prep.NormalizerTemporal()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    # Parse labels based on the task solved (artifact detection/sleep staging)
    scorings = []
    for scoring in os.listdir(datapath['scorings']):
        scorings.append(datapath['scorings']+scoring)
    if cfg.PROBLEM == "AD":
        labels = ScoringParser(scorings).get_binary_scorings()
    elif cfg.PROBLEM == "SS":
        labels = ScoringParser(scorings).get_4stage_scorings()
    # Make sure #labels is equal to #features
    assert(np.shape(features)[0]==len(labels))
    # Add neighbors to capture the context
    features = add_neighbors(features, num_neighbors)
    # Concatenate features and labels, and save the data
    dataset = np.hstack((features,labels))
    np.savetxt(datapath['csv'], dataset, delimiter=",")
    print "# The shape of training data is: " + str(np.shape(dataset))
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
# -- parse number of gpu if specified
if len(sys.argv)>2:
    gpu = sys.argv[2]
else:
    gpu = "0"

# -- parse command
command = sys.argv[1]
if command == 'prepare':
    prepare()
elif command == 'train':
    train(gpu)
else:
    print "Unknown Command!"
