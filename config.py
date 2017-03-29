# Author: Djordje Miladinovic
# License:

# ---------------------------------------------------------------------------- #
# ------------ User defined variables - to be set ---------------------------- #
# ---------------------------------------------------------------------------- #
# Currently we are able to solve two problems
# "AD" - artifact detection
# "SS" - sleep staging
PROBLEM = "SS"
# Number of classes
num_classes = 3 if PROBLEM == 'SS' else 2
# Optimization parameters
learning_rate = 0.0003
learning_rate_decay = 0.1
momentum = 0.5
dropout = 0.6
weight_decay = 0.002
batch_size = 1
max_epochs = 50
num_neighbors = 4 # should be even number (bi-directional)

# ---------------------------------------------------------------------------- #
# -- File paths to data sets: generally, there is no need to change this ----- #
# ---------------------------------------------------------------------------- #
TRAINSET = {"recordings" : "./data/recordings/train/",\
            "scorings" : "./data/scorings/train/",\
            "csv" : "./data/CSV/trainset.csv"}
HOLDOUT = {"recordings" : "./data/recordings/holdout/",\
           "scorings" : "./data/scorings/holdout/",\
           "csv" : "./data/CSV/holdout.csv"}
TESTSET = {"recordings" : "./data/recordings/test/",\
           "scorings" : "./data/scorings/test/",\
           "csv" : "./data/CSV/testset.csv"}
