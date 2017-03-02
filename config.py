"""
Global configuration parameters.
"""
# Author: Djordje Miladinovic
# License:


# ---------------------------------------------------------------------------- #
# ----------------- Encoding ------------------------------------------------- #
# ---------------------------------------------------------------------------- #
FORMAT = "UZH" # So far implemented are "UZH" and "USZ"
# Currently we are able to solve two problems
# "ART" - artifact detection
# "SS" - full sleep staging
PROBLEM_TYPE = "SS"
# The name of the corresponding neural network architecture
ARCHITECTURE = "temporal_convolution"+"_"+FORMAT+"_"+PROBLEM_TYPE
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
