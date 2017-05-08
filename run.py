# Author: Djordje Miladinovic
# License:

import config as cfg
import os
import sys
from sklearn.externals import joblib
import sslib.preprocessing as prep
import sslib.parsing as pars
import numpy as np
from sklearn.preprocessing import RobustScaler

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

    cpars = joblib.load('models/cpars.pkl')
    print "## Calibration parameters: " + str(cpars)

    # Extract temporal channels
    recordings = []
    for rcd in os.listdir(datapath['recordings']):
        if rcd[0]!='.': recordings.append(datapath['recordings']+rcd)
    assert recordings!=[], "No .edf recordings in corresponding folder!"
    data = FeatureExtractor(recordings, cpars)\
          .get_temporal_features(normalize=True, filtering=False)
 
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
    os.system('CUDA_VISIBLE_DEVICES='+gpu+' '+'th sslib/deepnet/test.lua'\
              +' -trainedModelPath '+str(cfg.TRAINED_MODEL_PATH)\
              +' -dataPath '+str(cfg.TESTSET['csv']))

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
              +' -trainedModelPath '+str(cfg.TRAINED_MODEL_PATH)\
              +' -dataPath '+str(cfg.EVALSET['csv'])\
              +' -predictionsPath '+str(cfg.PREDICTIONS))
    print "## Predictions generated!"

def dstats():
    """Get data set statistics"""
    w_all = 0
    n_all = 0
    r_all = 0
    a_all = 0
    u_all = 0
    for datapath in [cfg.TESTSET,cfg.TRAINSET]:#[cfg.TRAINSET]:#, cfg.HOLDOUT, cfg.TESTSET]:
        # Get recordings
        recordings = []
        for rcd in sorted(os.listdir(datapath['recordings'])):
            if rcd[0]!='.':
                print "------------------------------------"
                # temporal signals
                labels = ScoringParser(datapath['scorings']+rcd.split('.')[0]+'.STD').get_4stage_scorings()
                data = FeatureExtractor(datapath['recordings']+rcd, np.ones(num_channels)).get_temporal_features(normalize=True,filtering=False)
                # overall stats
                eeg1 = data[:,0:512].flatten()
                eeg2 = data[:,512:2*512].flatten()
                emg  = data[:,2*512:3*512].flatten()
               # print "EEG1: " + str(np.percentile(eeg1,90))#std = " + str(np.std(eeg1)) + " rs = " + str(RobustScaler().fit(eeg1.reshape(-1,1)).scale_)
               # print "EEG2: " + str(np.percentile(eeg2,90))#std = " + str(np.std(eeg2)) + " rs = " + str(RobustScaler().fit(eeg2.reshape(-1,1)).scale_)
               # print "EMG: " + str(np.percentile(emg,90))#std = " + str(np.std(emg)) + " rs = " + str(RobustScaler().fit(emg.reshape(-1,1)).scale_)
                # dump ambigious
                #data = data[labels.flatten()<cfg.num_classes+1,:]
                #labels = labels[labels.flatten()<cfg.num_classes+1]
                print np.shape(data)
                print np.shape(labels)
                w = len(labels[labels.flatten()==1])*1.0/len(labels)
                n = len(labels[labels.flatten()==2])*1.0/len(labels)
                r = len(labels[labels.flatten()==3])*1.0/len(labels)
                a = len(labels[labels.flatten()==4])*1.0/len(labels)
                u = len(labels[labels.flatten()==5])*1.0/len(labels)
                print "Stage distribution - Wake: "+str(w)+" NREM: "+str(n)+" REM:"+str(r)+" ART: "+str(a)+" AMB:"+str(u)

                # per-label stats
                for i in range(1,4):
                    eeg1 = data[labels.flatten()==i,0:512].flatten()
                    eeg2 = data[labels.flatten()==i,512:2*512].flatten()
                    emg  = data[labels.flatten()==i,2*512:3*512].flatten()
                    #print "Label = " + str(i)
                    #print "   std  = " + str(np.std(eeg1)) + " - " + str(np.std(eeg2)) + " - " + str(np.std(emg))
                w_all = w_all + w
                n_all = n_all + n
                r_all = r_all + r
                a_all = a_all + a
                u_all = u_all + u
    print w_all/16
    print n_all/16
    print r_all/16
    print a_all/16
    print u_all/16

def pstats():
    """Get prediction statistics"""

    a = np.genfromtxt(cfg.PREDICTIONS, skip_header=2, delimiter=',', dtype=int)
    print len(a)
    print len(a[a==1])
    print len(a[a==2])
    print len(a[a==3])


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
elif command == 'pstats':
    pstats()
elif command == 'dstats':
    dstats()
else:
    print "Unknown Command!"

