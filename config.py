# Author: Djordje Miladinovic
# License:

# ---------------------------------------------------------------------------- #
# ------------ User defined variables - to be set ---------------------------- #
# ---------------------------------------------------------------------------- #
# Currently we are able to solve two problems
# "AD" - artifact detection
# "SS" - sleep staging
PROBLEM = "SS"
num_classes = 3 if PROBLEM == 'SS' else 2

# Optimization parameters
learning_rate = 0.00005
learning_rate_decay = 0.1
momentum = 0.5
dropout = 0.5
weight_decay = 0.0001
batch_size = 100
max_epochs = 50
num_neighbors = 2 # must be even number (bi-directional)

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
