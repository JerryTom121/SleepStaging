"""Main script which demonstrates the usage of sslib.
"""
# Author: Djordje Miladinovic
# License:

# Set the Training flag
RETRAIN_MODEL = False

import config as cfg
import os
import subprocess
import numpy as np
from sklearn.externals import joblib
from sklearn import metrics
import sslib.preprocessing as prep
import sslib.parsing as pars


def to_be_removed(features):
    """Temporary function. Remove when possible.
    """
    return features[:,::2] 

def train_model():
    """Preform retraining using data given in /data folder. The model
    parameters will be saved in the corresponding folder.
    """

    print "# Debug: model is being retrained... "

    # Parse, normalize and save features
    scaler = prep.NormalizerTemporal()
    recordings = []
    for recording in os.listdir(cfg.PATH_TO_TRAIN_RECORDINGS):
        recordings.append(cfg.PATH_TO_TRAIN_RECORDINGS+recording)
    features = prep.FeatureExtractorUZH(recordings).get_temporal_features()
    features = scaler.fit_transform(features)
    print "# Debug feature matrix for training: " + str(np.shape(features))
    np.savetxt(cfg.PATH_TO_TRAIN_FEATURES, features, delimiter=",")

    # Parse and save labels
    scorings = []
    for scoring in os.listdir(cfg.PATH_TO_TRAIN_SCORINGS):
        scorings.append(cfg.PATH_TO_TRAIN_SCORINGS+scoring)
    labels = pars.ScoringParserUZH(scorings).get_binary_scorings()
    print "# Debug label array for training: " + str(np.shape(labels))
    np.savetxt(cfg.PATH_TO_TRAIN_LABELS, labels, delimiter=",")

    # TODO:
    # Call lua to train a model on above created files
    # save the model in cfg.PATH_TO_NNMODEL
    # ------------------------------------------------

    # Save scaler
    joblib.dump(scaler, cfg.PATH_TO_SCALER)




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
    scaler = joblib.load(cfg.PATH_TO_SCALER)
    features = prep.FeatureExtractorUZH(cfg.PATH_TO_TEST_RECORDINGS+recording)\
                   .get_temporal_features()
    features = scaler.transform(features)
    features = to_be_removed(features) # TO BE REMOVED !!!!!!!!!!!!!!!!!!!!!!!!!!!
    #np.savetxt(cfg.PATH_TO_CSV+recording+"_features.csv", features, delimiter=",")

    # Make predictions
    print(subprocess.check_output([#'CUDA_VISIBLE_DEVICES=0',\
                    'th',\
                    'sslib/deepnet/predict.lua',\
                    cfg.PATH_TO_NNMODEL,\
                    cfg.PATH_TO_CSV+recording+"_features.csv",\
                    cfg.PATH_TO_CSV+recording.split('.')[0]+"_preds.csv"]))
    
    # Remove feature file
    #os.remove(cfg.PATH_TO_CSV+recording+"_features.csv")

def evaluate(recording):
    """Evaluate prediction
    """
    preds = np.genfromtxt(cfg.PATH_TO_CSV+recording.split('.')[0]+"_preds.csv",\
                          skip_header=2,delimiter=',',dtype=int)

    truth = pars.ScoringParserUZH(cfg.PATH_TO_TEST_SCORINGS+\
                                   recording.split('.')[0]+".STD")\
                                   .get_binary_scorings()\
                                   .flatten()

    print "Confusion matrix: "
    print metrics.confusion_matrix(truth, preds)

# ---------------------------------------------------------------------------- #
# --------------- Cleanup folder for .csv files ------------------------------ #
# ---------------------------------------------------------------------------- #
#for file_ in os.listdir(cfg.PATH_TO_CSV):
#    os.remove(cfg.PATH_TO_CSV+file_)

# ---------------------------------------------------------------------------- #
# --------------- Train or load existing model ------------------------------- #
# ---------------------------------------------------------------------------- #
if RETRAIN_MODEL:
    train_model()

# ---------------------------------------------------------------------------- #
# --------------- Evaluate accuracy on each file from test-data folder ------- #
# ---------------------------------------------------------------------------- #
for recording in os.listdir(cfg.PATH_TO_TEST_RECORDINGS):
 #   predict(recording)
    evaluate(recording)
