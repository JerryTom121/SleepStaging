"""
Global configuration parameters.
"""
# Author: Djordje Miladinovic
# License:


# ---------------------------------------------------------------------------- #
# ------------ User defined variables - to be set ---------------------------- #
# ---------------------------------------------------------------------------- #
# So far implemented are "UZH" and "USZ"
FORMAT = "UZH"
# Currently we are able to solve two problems
# "ART" - artifact detection
# "SS" - full sleep staging
PROBLEM_TYPE = "SS"
# ---------------------------------------------------------------------------- #
# ------------ Model specification ------------------------------------------- #
# ---------------------------------------------------------------------------- #
# The name of the corresponding neural network architecture
ARCHITECTURE = "temporal_convolution"+"_"+FORMAT+"_"+PROBLEM_TYPE
# ---------------------------------------------------------------------------- #
# ------------ File paths - generally, not to be changed --------------------- #
# ---------------------------------------------------------------------------- #
# class
class DATASET(object):
    RECORDINGS = ""
    SCORINGS = ""
    CSV = ""
# Parsed raw data is saved in .csv format
CSV_HOME = "./data/CSV/"
# Training data set
TRAINSET = DATASET()
TRAINSET.RECORDINGS = "./data/recordings/train/"
TRAINSET.SCORINGS = "./data/scorings/train/"
TRAINSET.CSV = CSV_HOME+"trainset.csv"
# Holdout data set
HOLDOUT = DATASET()
HOLDOUT.RECORDINGS = "./data/recordings/holdout/"
HOLDOUT.SCORINGS = "./data/scorings/holdout/"
HOLDOUT.CSV = CSV_HOME+"holdout.csv"
# Validation data set
TESTSET = DATASET()
TESTSET.RECORDINGS = "./data/recordings/test/"
TESTSET.SCORINGS = "./data/scorings/test/"
TESTSET.CSV = CSV_HOME+"testset.csv"
# Folder containg models learned from training
MODELS_HOME = "./models/"
MODELS_SCALER = MODELS_HOME+"scaler.pkl"
MODELS_ARCHITECTURE = MODELS_HOME+ARCHITECTURE
