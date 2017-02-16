"""Main script which demonstrates the usage of sslib.
"""
# Author: Djordje Miladinovic
# License:

# Set the Training flag
RETRAIN_MODEL = False
EVALUATE_PREDICTIONS = False

import config as cfg
import os
from sklearn.externals import joblib
import sslib.preprocessing as prep
import sslib.parsing as pars


# ---------------------------------------------------------------------------- #
# --------------- Retrain the model if specified ----------------------------- #
# ---------------------------------------------------------------------------- #
if RETRAIN_MODEL:

    print "# Debug: model retraining specified: "

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

    # Save paramaters
    joblib.dump(scaler, 'models/scaler.pkl')
    # save(model)
else:

    print "# Debug: loading existing model: "

    # Load previously trained feature normalizer
    scaler = joblib.load('models/scaler.pkl')

    # bla bla bla
    # NYI: model = load()

# ---------------------------------------------------------------------------- #
# ------ Make and evaluate predictions on test recordings -------------------- #
# ---------------------------------------------------------------------------- #
for recording in os.listdir(cfg.PATH_TO_TEST_RECORDINGS):

    print "# Debug: processing recording " + recording
    features = prep.FeatureExtractorUZH(cfg.PATH_TO_TEST_RECORDINGS+recording)\
                                 .get_temporal_features()
    features = scaler.transform(features)

    print "# Debug: Making predictions..."

    # NYI: predictions = predict(model,features)

    if EVALUATE_PREDICTIONS:

        print "# Debug: Evaluating predictions:"

        # NYI: filename =
        # scorings = parsing.ScoringExtractorUZH\
        #           (PATH_TO_TEST_SCORINGS+filename).get_binary_scorings()
        # NYI: evaluate(predictions,scorings)
    else:

        print "# Debug: Saving predictions into " + cfg.PATH_TO_PREDICTIONS

        # NYI: save(predictions,cfg.PATH_TO_PREDICTIONS)
