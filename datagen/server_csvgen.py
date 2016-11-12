import numpy as np
import sys 
sys.path.insert(0, '../library')
from Experiment import Experiment 
from features import extract_features
# --------------------------------------------------------------------------- #
# -- Choose mapping based on the classification goal: ----------------------- #
# -- do we want to detect anomalies or classify sleeping stages ------------- #
# --------------------------------------------------------------------------- #
MAPPING = { "w": +1, "n": +1, "r": +1,                               # normal
            "1": -1, "2": -1, "3": -1, "a": -1, "'": -1, "4": -1 }   # artefact       
# --------------------------------------------------------------------------- #
# --------------------- Load Experiment ------------------------------------- #
# --------------------------------------------------------------------------- #
Exp1 = Experiment('1') # Experiment 1 is reserved for the server
Exp2 = Experiment('2') # Experiment 2 is reserved for the server
# --------------------------------------------------------------------------- #
# ----------------- Extract features and labels ----------------------------- #
# --------------------------------------------------------------------------- #
# part 1
[X,Y] = extract_features(Exp1,MAPPING,Exp1.testset)
idx_X = np.linspace(1, len(X), len(X)).astype(np.int);
data = np.c_[idx_Xt, Xt, Yt]
np.savetxt('../../generated_data/temporal_data.csv', data, fmt='%s', delimiter=',')
# part 2
[X,Y] = extract_features(Exp2,MAPPING,Exp2.testset)
idx_X = np.linspace(1, len(X), len(X)).astype(np.int);
data = np.c_[idx_Xt, Xt, Yt]
np.savetxt('../../generated_data/fourier_data.csv', data, fmt='%s', delimiter=',')
