# Author: Djordje Miladinovic
# License:

# ---------------------------------------------------------------------------- #
# -- Variables related to data preparation and feature extraction  ----------- #
# ---------------------------------------------------------------------------- #
# We have the following classes: {WAKE, REM, NREM, ARTIFACT}
# - in case we do not consider artifacts set this number to 3 else 4
num_classes = 3
# Number of neighbours to be included - must be even number (bi-directional)
num_neighbors = 2

# ---------------------------------------------------------------------------- #
# -- Variables related to training procedure --------------------------------- #
# ---------------------------------------------------------------------------- #
learning_rate = 0.00001
learning_rate_decay = 0.1
dropout = 0.5
weight_decay = 0.001
batch_size = 100
max_epochs = 50

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
