"""Main script which demonstrates the usage of sslib.
   arg {'prepare','train','evaluate'} 
"""
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

# ---------------------------------------------------------------------------- #
# ----- Configure EEG/EMG recording, and scoring file format ----------------- #
# ---------------------------------------------------------------------------- #
if cfg.FORMAT == "UZH":
    FeatureExtractor = prep.FeatureExtractorUZH
    ScoringParser = pars.ScoringParserUZH
    fextension = '.STD'
elif cfg.FORMAT == "USZ":
    FeatureExtractor = prep.FeatureExtractorUSZ
    ScoringParser = pars.ScoringParserUSZ
    fextension = '.txt'
else:
    print "Uknown encoding: " + cfg.FORMAT
    exit()

# ---------------------------------------------------------------------------- #
# ----- Utility functions ---------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def prepare():
    """
    """

    # Parse, normalize and save features
    scaler = prep.NormalizerTemporal()
    recordings = []
    for recording in os.listdir(cfg.PATH_TO_TRAIN_RECORDINGS):
        recordings.append(cfg.PATH_TO_TRAIN_RECORDINGS+recording)
    features = FeatureExtractor(recordings).get_temporal_features()
    features = scaler.fit_transform(features)
    
    # Parse and save labels
    scorings = []
    for scoring in os.listdir(cfg.PATH_TO_TRAIN_SCORINGS):
        scorings.append(cfg.PATH_TO_TRAIN_SCORINGS+scoring)
    if cfg.PROBLEM_TYPE == "ART":
        labels = ScoringParser(scorings).get_binary_scorings()
    elif cfg.PROBLEM_TYPE == "SS":
        labels = ScoringParser(scorings).get_4stage_scorings()
        
    # make sure we do not have more labels than features
    features = features[0:len(labels),:]

    # Clean CSV directory
    for file_ in os.listdir(cfg.PATH_TO_CSV):
        os.remove(cfg.PATH_TO_CSV+file_)

    # Concatenate features and labels, and save the data
    dataset = np.hstack((features,labels))
    print "# The shape of data is: " + str(np.shape(dataset))
    np.savetxt(cfg.PATH_TO_TRAINING, dataset, delimiter=",")

    # Save scaler
    joblib.dump(scaler, cfg.PATH_TO_SCALER)

def train():
    """Preform retraining using data given in /data folder. The model
    parameters will be saved in the corresponding folder.
    """

    print "# Debug: model is being retrained... "
    os.system('th sslib/deepnet/train.lua ' \
              + cfg.PATH_TO_TRAINING + ' '\
              + cfg.PATH_TO_MODELS + ' '\
              + cfg.ARCHITECTURE)

def predict(recording):
    """Make predictions on a given recording

    Parameters
    ----------
        recording: EEG/EMG recording on which we evaluate our model
    Returns
    -------
        Numpy array of predictions
    """
    
    # Fetch and transform features
    features = FeatureExtractor(cfg.PATH_TO_TEST_RECORDINGS+recording)\
               .get_temporal_features()
    scaler = joblib.load(cfg.PATH_TO_SCALER)
    features = scaler.transform(features)
    np.savetxt(cfg.PATH_TO_CSV+recording+"_features.csv", features, delimiter=",")

    # Make predictions
    os.system('th sslib/deepnet/predict.lua ' \
              + cfg.PATH_TO_NNMODEL + ' '\
              + cfg.PATH_TO_CSV + recording + '_features.csv '\
              + cfg.PATH_TO_CSV + recording.split('.')[0] + '_preds.csv')
    
    # Remove feature file
    os.remove(cfg.PATH_TO_CSV+recording+"_features.csv")

def evaluate(recording):
    """Evaluate prediction
    """

    # Read previously generated predictions
    preds = np.genfromtxt(cfg.PATH_TO_CSV+recording.split('.')[0]+"_preds.csv",\
                          skip_header=2,delimiter=',',dtype=int)

    # Read corresponding labels
    sparser = ScoringParser(cfg.PATH_TO_TEST_SCORINGS+\
                            recording.split('.')[0]+fextension)
    if cfg.PROBLEM_TYPE == "ART":
        truth = sparser.get_binary_scorings().flatten()
    elif cfg.PROBLEM_TYPE == "SS":
        truth = sparser.get_4stage_scorings().flatten()

    # Make sure we have number of predictions equal to number of labels
    preds = preds[0:len(truth)]

    print "Confusion matrix: "
    cmat = metrics.confusion_matrix(truth, preds)
    print "----------------------------------------"
    print "| EVAL: Artifact detection confusion matrix:"
    print cmat
    print "----------------------------------------"
    print "| EVAL: Artifact detection evaluation:"
    print "| Accuracy: " + format(metrics.accuracy_score(truth, preds, '.2f'))
    print "| Recall: " + format(cmat[0, 0]*1.0/(cmat[0, 0]+cmat[0, 1]), '.2f')
    print "| Precision: "+ format(cmat[0, 0]*1.0/(cmat[0, 0]+cmat[1, 0]), '.2f')
    print "----------------------------------------"

# ---------------------------------------------------------------------------- #
# - Parse command to process data, train a model or evaluate already trained - #
# ---------------------------------------------------------------------------- #
command = sys.argv[1]

if command == 'prepare':
    prepare()
elif command == 'train':
    train()
elif command == 'evaluate':
    # predict and evalute scorings for each file of the test folder
    for recording in os.listdir(cfg.PATH_TO_TEST_RECORDINGS):
        predict(recording)
        evaluate(recording)
else:
    print "Unknown command!"

