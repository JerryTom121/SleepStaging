"""Main script which demonstrates the usage of sslib.
"""
# Author: Djordje Miladinovic
# License:

# Set the Training flag
TRAINING = False

import config as cfg
import os
import subprocess
import numpy as np
from sklearn.externals import joblib
from sklearn import metrics
import sslib.preprocessing as prep
import sslib.parsing as pars


def prepare_data():
    """
    """

    # Parse, normalize and save features
    scaler = prep.NormalizerTemporal()
    recordings = []
    for recording in os.listdir(cfg.PATH_TO_TRAIN_RECORDINGS):
        recordings.append(cfg.PATH_TO_TRAIN_RECORDINGS+recording)
    features = prep.FeatureExtractorUZH(recordings).get_temporal_features()
    features = scaler.fit_transform(features)

    # Parse and save labels
    scorings = []
    for scoring in os.listdir(cfg.PATH_TO_TRAIN_SCORINGS):
        scorings.append(cfg.PATH_TO_TRAIN_SCORINGS+scoring)
    labels = pars.ScoringParserUZH(scorings).get_binary_scorings()

    # Concatenate features and labels, and save the data
    dataset = np.hstack((features,labels))
    print "# The shape of data is: " + str(np.shape(dataset))
    np.savetxt(cfg.PATH_TO_TRAINING, dataset, delimiter=",")

    # Save scaler
    joblib.dump(scaler, cfg.PATH_TO_SCALER)

def train_model():
    """Preform retraining using data given in /data folder. The model
    parameters will be saved in the corresponding folder.
    """

    print "# Debug: model is being retrained... "

    # Call lua to train a model on above created files
    print(subprocess.check_output([#'CUDA_VISIBLE_DEVICES=0',\
                    'th',\
                    'sslib/deepnet/train.lua',\
                    cfg.PATH_TO_TRAINING,\
                    cfg.PATH_TO_MODELS]))

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
    np.savetxt(cfg.PATH_TO_CSV+recording+"_features.csv", features, delimiter=",")

    # Make predictions
    print(subprocess.check_output([#'CUDA_VISIBLE_DEVICES=0',\
                    'th',\
                    'sslib/deepnet/predict.lua',\
                    cfg.PATH_TO_NNMODEL,\
                    cfg.PATH_TO_CSV+recording+"_features.csv",\
                    cfg.PATH_TO_CSV+recording.split('.')[0]+"_preds.csv"]))
    
    # Remove feature file
    os.remove(cfg.PATH_TO_CSV+recording+"_features.csv")

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
# ----- Retrain model or predict on existing, as specified ------------------- #
# ---------------------------------------------------------------------------- #

if TRAINING:
    # Retrain model
#    prepare_data()
    train_model()
else:
    # Clean CSV directory
    for file_ in os.listdir(cfg.PATH_TO_CSV):
        os.remove(cfg.PATH_TO_CSV+file_)
    # Predict and evaluate each test file
    for recording in os.listdir(cfg.PATH_TO_TEST_RECORDINGS):
        predict(recording)
        evaluate(recording)

