
# ---------------------------------------------------------------------------- #
# ------------------- Paths to folders and files ----------------------------- #
# ---------------------------------------------------------------------------- #
# Home folder for EEG/EMG recordings
EEG_FOLDER = "../data/EEGdata/"
# Home folder for .csv data
CSV_FOLDER = "../data/ProcessedData_CSV/"
# Home folder for results obtained from server
PREDS_FOLDER = "../data/Predictions_CSV/"
# Training .csv file for artifact detection
ARTDET_TRAIN_FILE = CSV_FOLDER + "training.csv"
# Testing .csv file for artifact detection
ARTDET_TEST_FILE = CSV_FOLDER + "testing.csv"
# ---------------------------------------------------------------------------- #
# ------------------- Pipeline configuration --------------------------------- #
# ---------------------------------------------------------------------------- #
shuffling = True
augmentation = True

