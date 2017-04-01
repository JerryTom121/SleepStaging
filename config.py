# Author: Djordje Miladinovic
# License:

# ---------------------------------------------------------------------------- #
# ------------ User defined variables - to be set ---------------------------- #
# ---------------------------------------------------------------------------- #
# Optimization parameters
learning_rate = 0.00001
learning_rate_decay = 0.1
dropout = 0.5
weight_decay = 0.0001
batch_size = 100
max_epochs = 50
num_neighbors = 2 # must be even number (bi-directional)

# ---------------------------------------------------------------------------- #
# -- Constants: generally, there is no need to change this ------------------- #
# ---------------------------------------------------------------------------- #
# we have the following classes: {wake, REM, NREM, Artifact}
num_classes = 4

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
