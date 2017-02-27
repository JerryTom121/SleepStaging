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


def prepare_data():
    """
    """

    # Parse, normalize and save features
    scaler = prep.NormalizerTemporal()
#    scaler = prep.NormalizerEnergy()
    recordings = []
    for recording in os.listdir(cfg.PATH_TO_TRAIN_RECORDINGS):
        recordings.append(cfg.PATH_TO_TRAIN_RECORDINGS+recording)

#    features = prep.FeatureExtractorUSZ(recordings).get_temporal_features() ######## new
#    features = prep.FeatureExtractorUSZ(recordings).get_energy_features() ######## new
#    features = prep.FeatureExtractorUZH(recordings).get_energy_features() ######## new
    features = prep.FeatureExtractorUZH(recordings).get_temporal_features()
    features = scaler.fit_transform(features)
    
    # Parse and save labels
    scorings = []
    for scoring in os.listdir(cfg.PATH_TO_TRAIN_SCORINGS):
        scorings.append(cfg.PATH_TO_TRAIN_SCORINGS+scoring)
#    labels = pars.ScoringParserUSZ(scorings).get_binary_scorings() ######### new
    labels = pars.ScoringParserUZH(scorings).get_binary_scorings()
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

def train_model():
    """Preform retraining using data given in /data folder. The model
    parameters will be saved in the corresponding folder.
    """

    print "# Debug: model is being retrained... "
    os.system('th sslib/deepnet/train.lua '+cfg.PATH_TO_TRAINING+' '\
                                           +cfg.PATH_TO_MODELS+' '\
                                           +cfg.ARCHITECTURE)
    #shtrain(cfg.PATH_TO_TRAINING)

def predict(recording):
    """Make predictions on a given recording

    Parameters
    ----------
        recording: EEG/EMG recording on which we evaluate our model

    Returns
    -------
        Numpy array of predictions

    """
    print recording
    # Fetch and transform features
    if recording.split('.')[0]=='Trial1' or recording.split('.')[0]=='Trial2':
        features = prep.FeatureExtractorUSZ(cfg.PATH_TO_TEST_RECORDINGS+recording)\
                   .get_temporal_features()
    else:
        features = prep.FeatureExtractorUZH(cfg.PATH_TO_TEST_RECORDINGS+recording)\
                       .get_temporal_features()
    scaler = joblib.load(cfg.PATH_TO_SCALER)
    features = scaler.transform(features)
    np.savetxt(cfg.PATH_TO_CSV+recording+"_features.csv", features, delimiter=",")

    # Make predictions
    os.system('th sslib/deepnet/predict.lua ' + cfg.PATH_TO_NNMODEL+' '\
                                              + cfg.PATH_TO_CSV+recording+'_features.csv '\
                                              + cfg.PATH_TO_CSV+recording.split('.')[0]+'_preds.csv')

    # Remove feature file
    os.remove(cfg.PATH_TO_CSV+recording+"_features.csv")

def evaluate(recording):
    """Evaluate prediction
    """
    preds = np.genfromtxt(cfg.PATH_TO_CSV+recording.split('.')[0]+"_preds.csv",\
                          skip_header=2,delimiter=',',dtype=int)

    
    if recording.split('.')[0]=='Trial1' or recording.split('.')[0]=='Trial2':
        truth = pars.ScoringParserUSZ(cfg.PATH_TO_TEST_SCORINGS+\
                                       recording.split('.')[0]+".txt")\
                                       .get_binary_scorings()\
                                       .flatten()
        if len(truth)<len(preds):
            preds = preds[0:len(truth)]
    else:
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
command = sys.argv[1]

if command == 'prepare':
    prepare_data()

elif command == 'train':
    train_model()

elif command == 'evaluate':
    # predict and evalute scorings for each file of the test folder
    for recording in os.listdir(cfg.PATH_TO_TEST_RECORDINGS):
        predict(recording)
        evaluate(recording)

else:
    print "Unknown command!"

