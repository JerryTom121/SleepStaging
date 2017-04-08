# Author: Djordje Miladinovic
# License:

import config as cfg
import os
import sys
from sklearn.externals import joblib
import sslib.preprocessing as prep
import sslib.parsing as pars
import numpy as np

# --------------------------------------------------------------------------- #
# -- UZH format specific initialization ------------------------------------- #
# --------------------------------------------------------------------------- #
# TODO: make configurable formatting
FeatureExtractor = prep.FeatureExtractorUZH
ScoringParser = pars.ScoringParserUZH
architecture = "temporal_convolution_UZH_SS"
signal_length = 512
num_channels = 3

# ---------------------------------------------------------------------------- #
# -- Auxiliary functions ----------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def add_neighbors(features, num_neighbors):
    """Extend feature matrix to surround each sample by num_neighbors/2 on front
    and num_neighbors/2 on the back."""

    # number of neighbors must be even (bi-directional context)
    assert(num_neighbors % 2 == 0)
    # initialize feature matrix
    F = np.array([])
    # surround each channel with its neighbors
    for c in range(num_channels):
        C = features[:, c*signal_length:(c+1)*signal_length]
        C_AUG = C.copy()
        for n in range(num_neighbors/2):
            PREV = np.roll(C, shift=(n+1), axis=0)
            SUBQ = np.roll(C, shift=-(n+1), axis=0)
            C_AUG = np.hstack((PREV, C_AUG, SUBQ))
        F = np.hstack((F, C_AUG)) if F.size else C_AUG
    # return newly constructed feature matrix
    return F

def generate_csv(datapath, scaler=None):
    """Given the path to data, parse raw recordings and scorings, merge these
    and write into a .csv file."""

    # Acquire recordings
    recordings = []
    for recording in os.listdir(datapath['recordings']):
        recordings.append(datapath['recordings']+recording)
    # Extract features
    #temporal_features = FeatureExtractor(recordings).get_temporal_features() # 3 channels
    fourier_features = FeatureExtractor(recordings).get_fourier_features() # 3 channels
    #features = np.hstack((temporal_features,fourier_features))
    #features = temporal_features
    features = fourier_features
    # Sanity check
    assert(np.shape(features)[1] == signal_length*num_channels)
    # Acquire scorings
    scorings = []
    for scoring in os.listdir(datapath['scorings']):
        scorings.append(datapath['scorings']+scoring)
    # Extract labels
    labels = ScoringParser(scorings).get_4stage_scorings()
    # Sanity check: #labels is equal to #features
    assert(np.shape(features)[0]==len(labels))
    # Leave only labeled samples used in the training process
    features = features[labels.flatten()<cfg.num_classes+1,:]
    labels = labels[labels.flatten()<cfg.num_classes+1]
    # Normalize features if scaler is given, otherwise make it
    if scaler==None:
        scaler = prep.NormalizerZMUV(num_channels, signal_length)
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
# -- Main functions ---------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def prepare():
    """ Given raw recordings and scorings file generate 3 different data sets in
    .csv format: training set, holdout set and testing set."""

    print "## Generate training data set:"
    scaler = generate_csv(cfg.TRAINSET)
    print "## Generate holdout data set:"
    generate_csv(cfg.HOLDOUT, scaler)
    print "## Generate test data set:"
    generate_csv(cfg.TESTSET, scaler)


def train(gpu):
    """Train model on specified GPU using previously prepared training and holdout
    data sets. Essentially we call corresponding .lua script in torch framework 
    and we use previously configured training parameters ('config.py' file)."""

    os.system('CUDA_VISIBLE_DEVICES='+gpu+' '+'th sslib/deepnet/train.lua'\
              +' -learningRate '+str(cfg.learning_rate)\
              +' -learningRateDecay '+str(cfg.learning_rate_decay)\
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
# -- Parse command to process data, train a model or evaluate already trained  #
# ---------------------------------------------------------------------------- #
command = sys.argv[1]
gpu = sys.argv[2] if len(sys.argv)>2 else "0"
if command == 'prepare':
    prepare()
elif command == 'train':
    train(gpu)
else:
    print "Unknown Command!"
