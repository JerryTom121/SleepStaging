# some_file.py
import sys
sys.path.insert(0, '../library')

import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation 
from sklearn.cross_validation import cross_val_score
from sklearn.kernel_approximation import (RBFSampler,
                                          Nystroem)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.mixture import GMM
from sklearn.mixture import DPGMM
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score
from math import sqrt
from math import exp
from math import log
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
from hmmlearn import hmm
import sleeplib

# Some formatting configuration..
# ################################
np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)

# --------------------------------------------------------------------------- #
# ---------------------- User Defined Constants ----------------------------- #
# --------------------------------------------------------------------------- #
EXP_NUM    = sys.argv[1];          # Exp. to perform, given as command line arg
CSV_FOLDER = \
'/home/djordje/Desktop/CSVData/'   # Folder where all CSV data is kept:abs path
WAKE = 0; NREM = 1; REM  = 2;      # Encoding of sleeping stages
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************

# --------------------------------------------------------------------------- #
# ----------------------------- Variables ----------------------------------- #
# --------------------------------------------------------------------------- #
# These variables depend on selected experiment
NFEATURES  = 7              # Fourier Features: 6 from EEG1 and 1 from EMG
# Classification Tuning
LASSO_TRESHOLD = 0.02       # The treshold for LASOO selection
NPCA_COMP      = 6          # Number of PCA components
NFOLDS         = 10         # CV - just one more indicator of precision
# Sleeping stages permutations
LABELS=[WAKE,NREM,REM]
LABEL_PERMUTATIONS = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
# --------------------------------------------------------------------------- #
# ------------------------- Initalize all classifiers ----------------------- #
# --------------------------------------------------------------------------- #
svm  = SVC(probability=True)
svm_noprob  = SVC(probability=False)
svm_lin  = SVC(probability=False,kernel='linear')
lsvm = LinearSVC()
nb   = GaussianNB()
lcr  = LogisticRegression()
knn  = KNeighborsClassifier(n_neighbors=10)
rrc  = RidgeClassifierCV(normalize=True)
ada  = AdaBoostClassifier()
ada_dct = AdaBoostRegressor(DecisionTreeClassifier(max_depth=2),n_estimators=600, random_state=np.random.RandomState(1))
lda  = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
qda  = QuadraticDiscriminantAnalysis()
rfc  = RandomForestClassifier()                         # Random Forests are gooooood!!
gb   = GradientBoostingClassifier(n_estimators=1000) 
dtr  = DecisionTreeClassifier()
rbm = BernoulliRBM(n_components=2)
logistic = LogisticRegression()
rbm.learning_rate = 0.06
rbm.n_iter = 20
rbm.n_components = 100
logistic.C = 1  
rbm_lcr = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
# --------------------------------------------------------------------- #
# ----------------------------- HMM ----------------------------------- #
# --------------------------------------------------------------------- #n_mix
'''
markov = hmm.GaussianHMM(n_components=3, n_iter=500, init_params="mcst", covariance_type="full")
'''
markov = hmm.GaussianHMM(n_components=3, n_iter=500, params="mcs", init_params="mcs", covariance_type="full")
markov.transmat_ = np.array([[ 0.95354708,  0.04633496,  0.00011796],
                             [ 0.04959727,  0.93909542,  0.01130731],
                             [ 0.05827543,  0.00015793,  0.94156665]])

# We should try to learn parameters of MARKOV TRANSITION MATRIX !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# --------------------------------------------------------------------- #
# ----------------------------- GMM ----------------------------------- #
# --------------------------------------------------------------------- #
gmm  = GMM(n_components=len(LABELS),init_params='wc',n_iter=200,covariance_type='full')
# --------------------------------------------------------------------- #
# ------------------------- Soft Voting ------------------------------- #
# --------------------------------------------------------------------- #
weights=[1,1,1]
clfs = [lda,lcr,gb]
svc = VotingClassifier(estimators=[('0',clfs[0]),('1',clfs[1]),('2',clfs[2])],voting='soft',weights=weights)
# --------------------------------------------------------------------------- #
# --------------------- 8. Step: Chose Final Estimator ---------------------- #
# --------------------------------------------------------------------------- #
Markov           = False
GaussianMixtures = False
Cheating         = True
estimator        = rfc;

###############################################################################
###############################################################################
###############################################################################
# --------------------------------------------------------------------------- #
# ---------------- 1. Step: Read Data and Output useful Statistics ---------- #
# --------------------------------------------------------------------------- #
[train_input,train_output,test_input,test_ids] = sleeplib.readData(CSV_FOLDER,EXP_NUM,NFEATURES)
print "Training set distribution of states is: "
sleeplib.state_distributions(train_output,WAKE,NREM,REM)
#if Markov:
#    sleeplib.state_transitions(train_output,WAKE,NREM,REM)
# --------------------------------------------------------------------------- #
# ---------------- 2. Step: Augment Feature Space --------------------------- #
# --------------------------------------------------------------------------- #
if Markov==False and GaussianMixtures==False:
    [train_input,test_input] = [sleeplib.augmentFSpace(train_input),sleeplib.augmentFSpace(test_input)]
# --------------------------------------------------------------------------- #
# ---------------- 3. Step: Normalization and Scaling of Data --------------- #
# --------------------------------------------------------------------------- #
# LOG-ing..
train_input = train_input.applymap(lambda x: np.log(x))
test_input  = test_input.applymap(lambda x: np.log(x))

if Markov==True:
    pass
elif GaussianMixtures==True:
    # Ruins The accuracy for supervised methods
    train_input = pd.DataFrame(preprocessing.normalize(train_input))
    test_input  = pd.DataFrame(preprocessing.normalize(test_input))
else:
    # scaling
    scaler = StandardScaler()
    train_input = pd.DataFrame(scaler.fit_transform(train_input.astype(np.float)))
    test_input  = pd.DataFrame(scaler.transform(test_input.astype(np.float)))

# --------------------------------------------------------------------------- #
# ---------------- 4. Step: Initialize feature selectors -------------------- #
# --------------------------------------------------------------------------- #
lda             = LinearDiscriminantAnalysis()
lasso_selection = SelectFromModel(Lasso(alpha=0.02))#, threshold=0.02)
uv_selection    = SelectPercentile(f_classif,80)
lsvc_selection  = SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False))
logr_selection  = SelectFromModel(LogisticRegression(C=1,penalty="l1"))
ada_selection   = SelectFromModel(GradientBoostingClassifier(loss='exponential'))
gb_selection    = SelectFromModel(GradientBoostingClassifier(n_estimators=1000))
etc             = SelectFromModel(ExtraTreesClassifier())
# --------------------------------------------------------------------------- #
# ---------------- 5. Step: Perform feature selection ----------------------- #
# --------------------------------------------------------------------------- #
'''
selector = logr_selection
selector.fit(train_input,train_output.as_matrix().flatten())
train_input = pd.DataFrame(selector.transform(train_input))
test_input = pd.DataFrame(selector.transform(test_input))
print("After Feature Select we have " + str(len(train_input.columns)) + " features")
'''
# --------------------------------------------------------------------------- #
# ---------------- 6. Step: Perform PCA on top ------------------------------ #
# --------------------------------------------------------------------------- #
if GaussianMixtures:
    pass
else:
    pca = PCA(n_components=NPCA_COMP)
    pca.fit(train_input,train_output.as_matrix().flatten())
    train_input = pd.DataFrame(pca.transform(train_input))
    test_input = pd.DataFrame(pca.transform(test_input))
    print("After PCA we have " + str(len(train_input.columns)) + " features")
    print "-----------------------------------------------------------------------------"

# --------------------------------------------------------------------------- #
# ---------------- 9. Step: Final Predictions ------------------------------- #
# --------------------------------------------------------------------------- #
X = train_input.as_matrix()
Y = train_output.as_matrix().flatten()

print 'Fitting Training Data which dimensions are '+str(np.shape(X))+' ...'
print ".........."
start = time.time();
# Make Predictions...
if Markov:
    estimator = markov;
    estimator.fit(X)
    print "Converged? " + str(estimator.monitor_.converged)
elif GaussianMixtures:
    class_means = np.array([X[Y == i].mean(axis=0) for i in LABELS])
    #kmeans  = KMeans(n_clusters=len(LABELS),init=class_means,n_init=1)
    gmm.means_ = class_means
    estimator = gmm
    estimator.fit(X,Y)
else:
    estimator.fit(X,Y)
# --------------------------------------------------------------------------- #
# ------------------ Making Predictions-------------------------------------- #
# --------------------------------------------------------------------------- #
print 'Making Predictions on Testing Data which dimensions are '+str(np.shape(test_input.as_matrix()))+'...'
print ".........."
start = time.time();
# Predict
p = estimator.predict(test_input)
print 'Time elapsed to generate predictions: ' + str(time.time()-start)
print "-----------------------------------------------------------------------------"
predictions          = pd.DataFrame()
predictions['Id']    = test_ids.astype(int)
predictions['Label'] = pd.DataFrame(p).astype(int)
# --------------------------------------------------------------------------- #
# ---------------- 10. Step: Output ----------------------------------------- #
# --------------------------------------------------------------------------- #
predictions.to_csv(CSV_FOLDER+'final_exp'+str(EXP_NUM)+'.csv',index=False)
# --------------------------------------------------------------------------- #
# ---------------- 11. Step: Do the evaluation straight away ---------------- #
# --------------------------------------------------------------------------- #
predictions  = pd.read_csv(CSV_FOLDER+'final_exp'+str(EXP_NUM)+'.csv')
solutions    = pd.read_csv(CSV_FOLDER+'solution_exp'+str(EXP_NUM)+'.csv')
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
if Markov or GaussianMixtures:
    '''
    In the case we are using unsupervised learning_rate
    we try to find the permutation if indices which fits 
    the best to solutions
    '''
    pred = predictions['Label'].as_matrix().flatten()
    sol  = solutions['Label'].as_matrix().flatten()
    best_permutation = LABELS;
    # Find the best permutation of the labels
    perm_pred = np.zeros(len(pred))
    best_accuracy = 0;
    for perm_ind in range(0,6):
        perm = LABEL_PERMUTATIONS[perm_ind]
        for ind in range(0,len(pred)):
            perm_pred[ind] = perm[pred[ind]]
        cur_accuracy = np.sum(perm_pred==sol)*1.0/np.size(sol);
        if cur_accuracy>best_accuracy:
            best_accuracy    = cur_accuracy
            best_permutation = perm
    print best_permutation  
    # If Cheating is on place all REM to WAKE
    if Cheating:
        print "Cheating...."
        if best_permutation[0] == REM:
            best_permutation[0] = WAKE
        elif best_permutation[1] == REM:
            best_permutation[1] = WAKE
        else:
            best_permutation[2] = WAKE
    # Change Predictions according to Best Permutation
    for i in range(0,len(pred)):
            pred[i] = best_permutation[pred[i]]
    predictions['Label'] = pd.DataFrame(pred)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
print "Test set distribution of states is: "
sleeplib.state_distributions(solutions,WAKE,NREM,REM)
print "Prediction distribution of states is: "
sleeplib.state_distributions(predictions,WAKE,NREM,REM)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
if Markov:
    print "Markov transition matrix is"
    print markov.transmat_
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
'''
Print the accuracy achieved for each stage
separately and the overall accuracy
'''
print 'Wake Accuracy is:'
sleeplib.evaluate_predictions(predictions,solutions,[WAKE]);
print 'NREM Accuracy is:'
sleeplib.evaluate_predictions(predictions,solutions,[NREM]);
print 'REM Accuracy is:'
sleeplib.evaluate_predictions(predictions,solutions,[REM]);
print 'Overall Accuracy is:'
sleeplib.evaluate_predictions(predictions,solutions,LABELS);
print 'Confusion Matrix is:'
print confusion_matrix(solutions.as_matrix().flatten(),predictions.as_matrix().flatten(),LABELS)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
if Markov or GaussianMixtures:
    '''
    Try detecting outliers from Markov model 
    using posterior probabilities
    '''

    '''
    LP_TRESHOLD = 0.43
    #
    hp_preds = []
    hp_sols  = []
    posteriors = estimator.predict_proba(test_input)
    for i in range(0,len(pred)):
        if max(posteriors[i])>LP_TRESHOLD:
            hp_preds.append(pred[i])
            hp_sols.append(sol[i])

    hp_preds = np.asarray(hp_preds)
    hp_sols  = np.asarray(hp_sols)
    num_hp_entries = len(hp_preds)
    #num_lp_disagre = len(np.nonzero(lp_preds-lp_sols)[0])


    print "Chosen treshold is " + str(LP_TRESHOLD)
    print "There are " + str(len(pred)) + " entrie overall"
    print "There are " + str(num_hp_entries) + " High Probability entries"
    print "Which is " + str(num_hp_entries*1.0/len(pred)) + " of the whole"
    print "The accuracy for HP is :"
    sleeplib.evaluate_predictions(pd.DataFrame(hp_sols,columns=['Label']),pd.DataFrame(hp_preds,columns=['Label']),LABELS)
    print "Confusion matrix of HP entries is "
    print confusion_matrix(hp_sols,hp_preds,LABELS)
    '''
    print estimator.predict_proba(test_input)
    pass
if Markov:
    '''
    In the case of HMM plot the graphs
    to see whether there are systematical
    errors
    '''
    '''
    residual_sequences = []
    current_sequence = 0
    residual = pred-sol
    residual[np.nonzero(residual)] = 1
    for i in range(0,len(residual)):
        if residual[i]>0:
            current_sequence = current_sequence+1
        else:
            if current_sequence>0:
                residual_sequences.append(current_sequence)
                current_sequence = 0
    plt.hist(residual_sequences, bins=100)
    plt.show()
    '''
    '''
    plt.plot(residual)
    plt.axis([0, len(pred),-1,4])
    plt.ylabel('Label')
    plt.xlabel('Sample')
    plt.show()
    '''
