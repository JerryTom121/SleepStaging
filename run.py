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

def generate_csv(datapath, evaluation=False):
    """Given the path to data, parse raw recordings and scorings, merge these
    and write into a .csv file."""

    # Acquire input adaptation parameters
    cpars = joblib.load('models/cpars.pkl')
    print "## Calibration parameters: " + str(cpars)

    # Extract temporal channels
    recordings = []
    for rcd in os.listdir(datapath['recordings']):
        if rcd[0]!='.': recordings.append(datapath['recordings']+rcd)
    assert recordings!=[], "No .edf recordings in corresponding folder!"
    data = FeatureExtractor(recordings, cpars).get_temporal_features()

    if not evaluation:
        # Extract corresponding labels if specified
        scorings = []
        for scr in os.listdir(datapath['scorings']):
            if scr[0]!='.': scorings.append(datapath['scorings']+scr)
        labels = ScoringParser(scorings).get_4stage_scorings()
        # Leave out ambigious labels
        data = data[labels.flatten()<cfg.num_classes+1,:] 
        labels = labels[labels.flatten()<cfg.num_classes+1]

    # Sanity check - #samples = #labels 
    assert(np.shape(data)[1] == signal_length*num_channels)
    
    # Add neighbors to capture the context
    data = add_neighbors(data, cfg.num_neighbors)
    
    # Add labels to the data if specified
    if not evaluation:
        data = np.hstack((data,labels))

    # Concatenate features and labels, and save the data
    print "## Storing data into a .csv file..."
    np.savetxt(datapath['csv'], data, delimiter=",")
    print "## The shape of stored data is: " + str(np.shape(data))

def reset_calibration():
    """Set all calibration parameters to 1"""

    cpars = np.ones(num_channels)
    joblib.dump(cpars, 'models/cpars.pkl') 


# ---------------------------------------------------------------------------- #
# -- Main functions ---------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def prepare():
    """ Given raw recordings and scorings file generate 3 different data sets in
    .csv format: training set, holdout set and testing set."""
    
    print "## Calibration parameters resetted"
    reset_calibration()
    print "## Parsing recordings and labels of the training data set.."
    generate_csv(cfg.TRAINSET)
    print "## Parsing recordings and labels of the holdout data set.."
    generate_csv(cfg.HOLDOUT)

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

def test(gpu):
    """Evaluate the quality of the trained model on a separate, 
    validation data set"""

    print "## Parsing recordings and labels of the validation data set.."
    generate_csv(cfg.TESTSET)
    print "## Making and evaluating predictions.."
    #os.system('CUDA_VISIBLE_DEVICES='+gpu+' '+'th sslib/deepnet/predict.lua'\
    #          +' -trainedModelPath '+str(cfg.TRAINED_MODEL_PATH)\
    #          +' -testPath '+str(cfg.TESTSET['csv'])\
    #          +' -predictionsPath '+str(cfg.PREDICTIONS))
    print "## NOT YET IMPLEMENTED!!!"

def calibrate(gpu):
    """Use calibration data to learn calibration parameters"""

    print "## Parsing recordings and labels of calibration data set.."
    generate_csv(cfg.CALIBRATION)
    print "## Infering calibration parameters.."
    os.system('CUDA_VISIBLE_DEVICES='+gpu+' '+'th sslib/deepnet/calibrate.lua'\
              +' -trainedModelPath '+str(cfg.TRAINED_MODEL_PATH)\
              +' -inputSize '+str(signal_length*(1+cfg.num_neighbors))\
              +' -numChannels '+str(num_channels)\
              +' -calibrationPath '+str(cfg.CALIBRATION['csv']))

def predict(gpu):
    """Make predictions on novel data"""

    print "## Parsing raw .edf recordings of uploaded data set.."
    generate_csv(cfg.EVALSET, True)
    print "## Making predictions.."
    os.system('CUDA_VISIBLE_DEVICES='+gpu+' '+'th sslib/deepnet/predict.lua'\
              +' -trainedModelPath '+str(cfg.FULL_MODEL_PATH)\
              +' -dataPath '+str(cfg.EVALSET['csv'])\
              +' -predictionsPath '+str(cfg.PREDICTIONS))
    print "## Predictions generated!"

# ---------------------------------------------------------------------------- #
# -- Parse command ----------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
command = sys.argv[1]
gpu = sys.argv[2] if len(sys.argv)>2 else "0"

if command == 'prepare':
    prepare()
elif command == 'train':
    train(gpu)
elif command == 'test':
    test(gpu)
elif command == 'calibrate':
    calibrate(gpu)
elif command == 'predict':
    predict(gpu)
else:
    print "Unknown Command!"

