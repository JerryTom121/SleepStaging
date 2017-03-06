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
    ext = '.STD'
elif cfg.FORMAT == "USZ":
    FeatureExtractor = prep.FeatureExtractorUSZ
    ScoringParser = pars.ScoringParserUSZ
    ext = '.txt'
else:
    print "Uknown encoding: " + cfg.FORMAT
    exit()


# ---------------------------------------------------------------------------- #
# ----- Utility functions ---------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def generate_csv(datapath, scaler=None):
    """Parse raw data from specified folder, scale it using a given scaler, and
    finally save it in .csv format.
    """

    # Parse features from raw data files
    recordings = []
    for recording in os.listdir(datapath.RECORDINGS):
        recordings.append(datapath.RECORDINGS+recording)
    features = FeatureExtractor(recordings).get_temporal_features()

    # Normalize if scaler is given, otherwise make it
    if scaler==None:
        scaler = prep.NormalizerTemporal()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    # Parse labels based on the task solved (artifact detection/sleep staging)
    scorings = []
    for scoring in os.listdir(datapath.SCORINGS):
        scorings.append(datapath.SCORINGS+scoring)
    if cfg.PROBLEM_TYPE == "ART":
        labels = ScoringParser(scorings).get_binary_scorings()
    elif cfg.PROBLEM_TYPE == "SS":
        labels = ScoringParser(scorings).get_4stage_scorings()

    # Make sure we do not have more labels than features (just in case..)
    features = features[0:len(labels),:]

    # Concatenate features and labels, and save the data
    dataset = np.hstack((features,labels))
    np.savetxt(datapath.CSV, dataset, delimiter=",")
    print "# The shape of training data is: " + str(np.shape(dataset))

    # Return scaler
    return scaler


def validate(truth, preds):
    """
    """

    cmat = metrics.confusion_matrix(truth, preds)
    print "----------------------------------------"
    print "| EVAL: Confusion matrix:"
    print cmat
    print "----------------------------------------"
    if cfg.PROBLEM_TYPE == "ART":
        print "| EVAL: Artifact detection evaluation:"
        print "| Accuracy: " + format(metrics.accuracy_score(truth[truth<3], preds[truth<3], '.2f'))
        print "| Recall: " + format(cmat[0, 0]*1.0/(cmat[0, 0]+cmat[0, 1]), '.2f')
        print "| Precision: "+ format(cmat[0, 0]*1.0/(cmat[0, 0]+cmat[1, 0]), '.2f')
        print "----------------------------------------"
    elif cfg.PROBLEM_TYPE == "SS":
        print "| EVAL: Sleep staging:"
        print "| Accuracy: " + format(metrics.accuracy_score(truth[truth<4], preds[truth<4], '.2f'))
        print "----------------------------------------"

# ---------------------------------------------------------------------------- #
# ----- Main functions ------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
def prepare():
    """Parse raw data of training and holdout data sets, and save them in .csv
    format
    """
    
    # Clean CSV directory
    for file_ in os.listdir(cfg.CSV_HOME):
        os.remove(cfg.CSV_HOME+file_)

    # Generate training data set
    scaler = generate_csv(cfg.TRAINSET)
    joblib.dump(scaler, cfg.MODELS_SCALER)

    # Generate holdout data set
    generate_csv(cfg.HOLDOUT, scaler)

def train(gpu):
    """Train model on the specified GPU machine, and using previously prepared
    training and holdout data sets
    """

    # Call torch to train the model
    os.system('CUDA_VISIBLE_DEVICES='+gpu+' '+'th sslib/deepnet/train.lua '\
              +cfg.TRAINSET.CSV+' '+cfg.HOLDOUT.CSV+' '+cfg.MODELS_HOME+' '\
              +cfg.ARCHITECTURE+' '+gpu)

def predict(recording, gpu):
    """Make predictions on a given raw recording from the validation data set
    """

    # Extract features
    features = FeatureExtractor(cfg.TESTSET.RECORDINGS+recording)\
               .get_temporal_features()

    # Scale features
    scaler = joblib.load(cfg.MODELS_SCALER)
    features = scaler.transform(features)

    # Temporarily save features in a feature file
    np.savetxt(cfg.CSV_HOME+recording+"_fts.csv", features, delimiter=",")

    # Make predictions on the feature file
    os.system('CUDA_VISIBLE_DEVICES='+gpu+' '+'th sslib/deepnet/predict.lua '\
              +cfg.MODELS_ARCHITECTURE+'_'+gpu+' '\
              +cfg.CSV_HOME+recording+'_fts.csv '\
              +cfg.CSV_HOME+recording.split('.')[0]+'_preds.csv')
    
    # Remove feature file
    os.remove(cfg.CSV_HOME+recording+"_fts.csv")

def evaluate(recording):
    """Evaluate prediction
    """

    # Read previously generated predictions
    preds = np.genfromtxt(cfg.CSV_HOME+recording.split('.')[0]+"_preds.csv",\
                          skip_header=2, delimiter=',', dtype=int)

    # Read corresponding labels based on the problem solved
    sparser = ScoringParser(cfg.TESTSET.SCORINGS+recording.split('.')[0]+ext)
    if cfg.PROBLEM_TYPE == "ART":
        truth = sparser.get_binary_scorings().flatten()
    elif cfg.PROBLEM_TYPE == "SS":
        truth = sparser.get_4stage_scorings().flatten()

    # Make sure we have number of predictions equal to number of labels
    preds = preds[0:len(truth)]

    # Output validation results
    validate(truth, preds)

# ---------------------------------------------------------------------------- #
# - Parse command to process data, train a model or evaluate already trained - #
# ---------------------------------------------------------------------------- #
command = sys.argv[1]
if len(sys.argv)>2:
    gpu = sys.argv[2]
else:
    gpu = "0"

if command == 'prepare':
    prepare()
elif command == 'train':
    train(gpu)
elif command == 'predict':
    # predict scorings for each file of the test folder
    for recording in os.listdir(cfg.TESTSET.RECORDINGS):
        predict(recording, gpu)
elif command == 'validate':
    # predict and evalute scorings for each file of the test folder
    for recording in os.listdir(cfg.TESTSET.RECORDINGS):
        predict(recording, gpu)
        evaluate(recording)
else:
    print "Unknown Command!"
