"""
Global configuration parameters.
"""
# Author: Djordje Miladinovic
# License:



ARCHITECTURE = "temporal_convolution"
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
PATH_TO_TRAINING = PATH_TO_CSV + "training.csv"
# Trained models
PATH_TO_MODELS = "./models/"
PATH_TO_SCALER = PATH_TO_MODELS + "scaler.pkl"
PATH_TO_NNMODEL = PATH_TO_MODELS + ARCHITECTURE
# ---------------------------------------------------------------------------- #
# ------------------ Deep learning configuration ----------------------------- #
# ---------------------------------------------------------------------------- #




# ---------------------------------------------------------------------------- #
# ------------------- Pipeline configuration --------------------------------- #
# ---------------------------------------------------------------------------- #
SHUFFLING = True
AUGMENTATION = True

