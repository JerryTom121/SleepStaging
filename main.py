"""Main script which demonstrates the usage of sslib.
"""
# Author: Djordje Miladinovic
# License:

# Set the Training flag
RETRAIN_MODEL = True

import config as cfg
import os
from sklearn.externals import joblib
import sslib.preprocessing as prep
import sslib.parsing as pars


def train_model():
    """Preform retraining using data given in /data folder.

    Returns
    -------
       scaler: a parameter for feature scaling
       model: trained neural network
    """

    # Fetch and normalize features
    scaler = prep.NormalizerTemporal()
    recordings = []
    for recording in os.listdir(cfg.PATH_TO_TRAIN_RECORDINGS):
        recordings.append(cfg.PATH_TO_TRAIN_RECORDINGS+recording)
    features = prep.FeatureExtractorUZH(recordings).get_temporal_features()
    features = scaler.fit_transform(features)

    # Fetch labels
    scorings = []
    for scoring in os.listdir(cfg.PATH_TO_TRAIN_SCORINGS):
        scorings.append(cfg.PATH_TO_TRAIN_SCORINGS+scoring)
    labels = pars.ScoringParserUZH(scorings).get_binary_scorings()

    # Preform training
    # trainer = Trainer(cfg.OPT_PARS,features,labels)
    # model = trainer.train()

    return scaler

def save_model(scaler, nnmodel):
    """Save parameters, usually done after training is preformed.

    Parameters
    ----------
        scaler: feature scaling set of parameters
        nnmodel: neural network model
    """

    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    # save(model)

def load_model():
    """Load existing model.

    Returns
    -------
        [scaler,nnmodel]: a tuple consisted of a scaler and a trained n. network
    """

    return joblib.load('models/scaler.pkl')

def make_predictions(recording, scaler, nnmodel):
    """Make predictions on a given recording

    Parameters
    ----------
        recording: EEG/EMG recording on which we evaluate our model

    Returns
    -------
        Numpy array of predictions

    """

    features = prep.FeatureExtractorUZH(cfg.PATH_TO_TEST_RECORDINGS+recording)\
                   .get_temporal_features()
    features = scaler.transform(features)

    # do something with nnmodel

    return None

def evaluate_and_save_predictions(recording, predictions):
    """Evaluate predictions on ground truth.
    NYI
    """
    pass


# ---------------------------------------------------------------------------- #
# --------------- Main part of the script ------------------------------------ #
# ---------------------------------------------------------------------------- #
if RETRAIN_MODEL:
    print "# Debug: preform training "
    scaler = train_model()
    save_model(scaler, None)
else:
    print "# Debug: loading existing model "
    scaler = load_model()

# for each test file make predictions and evaluate the quality
for recording in os.listdir(cfg.PATH_TO_TEST_RECORDINGS):
    predictions = make_predictions(recording, scaler, None)
    evaluate_and_save_predictions(recording, predictions)
