"""
Global configuration parameters.

ASSUMPTIONS:
	(1) Recordings are named: 'EEG1','EEG2','EMG'
	(2) Interval/epoch lasts 4 seconds - default argument of the parser
	(3) EEG1 is frontal and EEG2 is parietal (or the opposite?)
	(4) Files containing scorings are named the same as the ones containing raw
	files
        (5) File names do not contain dots

TODO:
	* Implement fourier and energy feature extractors in 
	  preprocessing/feature_extraction.py
	* Better normalization?
	* Introduce filtering (e.g. low pass)?

MAKE_SURE:
	* That the labels correspond to features.

"""
# Author: Djordje Miladinovic
# License:


# ---------------------------------------------------------------------------- #
# ------------------- File paths --------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Raw data
PATH_TO_TRAIN_RECORDINGS = "./data/recordings/train/"
PATH_TO_TEST_RECORDINGS = "./data/recordings/test/"
# Scorings
PATH_TO_TRAIN_SCORINGS = "./data/scorings/train/"
PATH_TO_TEST_SCORINGS = "./data/scorings/test/"
# CSV folder
PATH_TO_CSV = "./data/CSV/"
PATH_TO_TRAIN_FEATURES = PATH_TO_CSV + "train_features.csv"
PATH_TO_TRAIN_LABELS = PATH_TO_CSV + "train_labels.csv"
# Trained models
PATH_TO_MODELS = "./models/"
PATH_TO_SCALER = PATH_TO_MODELS + "scaler.pkl"
PATH_TO_NNMODEL = PATH_TO_MODELS + "model.net"
# ---------------------------------------------------------------------------- #
# ------------------ Deep learning configuration ----------------------------- #
# ---------------------------------------------------------------------------- #




# ---------------------------------------------------------------------------- #
# ------------------- Pipeline configuration --------------------------------- #
# ---------------------------------------------------------------------------- #
SHUFFLING = True
AUGMENTATION = True

