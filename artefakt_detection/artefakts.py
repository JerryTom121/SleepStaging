"""
Script used for testing the artefakt identification.
Calls functions from 'artlib' which try to detect artefakts.
"""
import sys; sys.path.insert(0, '../library')
from edfplus import load_edf 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# sklearn stuff
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC,OneClassSVM
from sklearn import preprocessing
# hmm library
from hmmlearn import hmm
# signal processing stuff
from scipy import signal
# my stuff
import artlib
import sleeplib
import features

# Some formatting configuration..
# ################################
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# --------------------------------------------------------------------------- #
# ---------------------- User Defined Constants ----------------------------- #
# --------------------------------------------------------------------------- #
CSV_FOLDER = \
'/home/djordje/Desktop/CSVData/'   # Folder where all CSV data is kept:abs path
WAKE = 0; NREM = 1; REM  = 2;      # Encoding of main sleeping stages
WART = 3; NART = 4; RART = 5;      # Encoding of artefakts
STAGES = [WAKE,NREM,REM,WART,NART,RART];
mapping = {                        # Mapping labels to identifiers
    'w': WAKE,
    'n': NREM,
    'r': REM,
    '1': WART, # Wake artefakt
    '2': NART, # NREM artefakt
    '3': RART, # REM  artefakt
};
EXP_NUM    = sys.argv[1];          # Exp. to perform, given as command line arg
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
'''
Try learning approach to 
distinguish artefakts !!!
'''
# Get data based on the current exeperiment
[X_train,Y_train,X_test,Y_test] = sleeplib.read_data(CSV_FOLDER,EXP_NUM)

# Loging
'''
X_train = np.log(X_train)
X_test  = np.log(X_test)
'''

'''
plt.plot(X_train[:,1],c='red')
plt.plot(X_train[:,2],c='green')
plt.plot(X_train[:,3],c='purple')
plt.show()
'''

'''
thr = 4
new_thr = 5.6

before_train = X_train[:,0]
a = np.zeros(np.shape(X_train))
a = np.vstack((a,X_test))
before_test  = a[:,0]
'''
A = X_train
B = X_test

eeg1_train = X_train.flatten()
eeg1_test  = X_test.flatten()

X_train = np.zeros((np.shape(X_train)[0],2))
X_test = np.zeros((np.shape(X_test)[0],2))

#arts = np.zeros(np.shape(X_train)[0])

k = 100
for i in range(k+2,eeg1_train.size-k-2):
    ind = i/512
    abs_mean_diff = np.abs(np.mean(eeg1_train[i-k:i-1])-np.mean(eeg1_train[i+1:i+k]))
    std_dev = np.std(eeg1_train[i-k:i+k])
    X_train[ind][0] = max(X_train[ind][0],abs_mean_diff)
    X_train[ind][1] = max(X_train[ind][1],std_dev)


k = 100
for i in range(k+2,eeg1_test.size-k-2):
    ind = i/512
    abs_mean_diff = np.abs(np.mean(eeg1_test[i-k:i-1])-np.mean(eeg1_test[i+1:i+k]))
    std_dev = np.std(eeg1_test[i-k:i+k])
    X_test[ind][0] = np.max(np.array([X_test[ind][0],abs_mean_diff]))
    X_test[ind][1] = np.max(np.array([X_test[ind][1],std_dev]))


# Scaling
scaler = preprocessing.StandardScaler()
#scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X_train.astype(np.float))
X_test  = scaler.transform(X_test.astype(np.float))
'''
before_thr = scaler.scale_[0]*thr
c = np.ones(np.shape(a))*before_thr
plt.plot(before_train,c='red')
plt.plot(before_test,c='green')
plt.plot(c,c='black')
plt.show()
'''

'''
# Experiment with threshold
print "Thresholding from the perspective of training data"
print "--------------------------------------------------"
print np.shape(X_train)
print "There are " + str(np.sum(X_train[:,0]>thr)) +  " samples having EMG amplitude above "+ str(thr) +" threshold"
print "There are " + str(np.sum(X_train[:,0]<=thr)) + " samples having EMG amplitude under "+ str(thr) +" threshold"
print "Number of hits is: " + str(np.sum((X_train[:,0]>thr) & (Y_train==-1)))
print "Number of false hits is: " + str(np.sum((X_train[:,0]>thr) & (Y_train==1)))

print "Thresholding from the perspective of testing data"
print "--------------------------------------------------"
print np.shape(X_test)
print "There are " + str(np.sum(X_test[:,0]>thr)) +  " samples having EMG amplitude above "+ str(thr) +" threshold"
print "There are " + str(np.sum(X_test[:,0]<=thr)) + " samples having EMG amplitude under "+ str(thr) +" threshold"
print "Number of hits is: " + str(np.sum((X_test[:,0]>thr) & (Y_test==-1)))
print "Number of false hits is: " + str(np.sum((X_test[:,0]>thr) & (Y_test==1)))
'''

'''
# Graphs before thresholding
plt.plot(X_train[:,0],c='red')
a = np.vstack((np.zeros(np.shape(X_train)),X_test))
plt.plot(a[:,0],c='green')
c = np.ones(np.shape(a))*thr
plt.plot(c,c='black')
c = np.ones(np.shape(a))*new_thr
plt.plot(c,c='blue')
plt.show()
# Graphs after thresholding
X_train_after = X_train[(X_train[:,0]<thr),:]
X_test_after = X_test[(X_test[:,0]<new_thr),:]
#
plt.plot(X_train_after[:,0],c='red')
a = np.vstack((np.zeros(np.shape(X_train_after)),X_test_after))
plt.plot(a[:,0],c='green')
c = np.ones(np.shape(a))*thr
plt.plot(c,c='black')
c = np.ones(np.shape(a))*new_thr
plt.plot(c,c='blue')
plt.show()
'''
'''
print np.shape(X_train)
print np.sum(Y_train==1)
print np.sum(Y_train==-1)
print np.shape(X_test)
print np.sum(Y_test==1)
print np.sum(Y_test==-1)

plt.plot(Y_train,c='green')
plt.plot(Y_test+4,c='red')
plt.show()
'''

'''
samp_freq = 128.0
for i in range(np.shape(A)[0]):
    if Y_train[i] == -1:
        # get signal
        sig = A[i]
        # plot it

        plt.plot(sig)

        # design filter
        #(N, Wn) = signal.buttord(wp=0.1, ws=0.3, gpass=3, gstop=40)
        #(N, Wn) = signal.buttord(wp=20.0/samp_freq/2, ws=20.0/samp_freq, gpass=3, gstop=40)
        #(b, a) = signal.butter(N, Wn)
        #sig = signal.lfilter(b, a, sig)

        #filt_sig = sig - signal.medfilt(sig,kernel_size=101)
        filt_sig = signal.medfilt(sig,kernel_size=51)
        #filt_sig = np.gradient(sig)
        plt.plot(filt_sig)

        print str(i) + ": " + str(X_train[i])
        # show plot
        plt.show()


for i in range(np.shape(B)[0]):
    if Y_test[i] == -1:

        sig = B[i]

        print i
        plt.plot(sig)

        #(N, Wn) = signal.buttord(wp=20.0/128/2, ws=20.0/128, gpass=3, gstop=40)
        #(b, a) = signal.butter(N, Wn)
        #sig = signal.lfilter(b, a, sig)

        sig = sig - signal.medfilt(sig,kernel_size=5)
        #sig = signal.medfilt(sig,kernel_size=51)
        plt.plot(sig+30)

        print str(i) + ": " + str(X_test[i])

        plt.show()
'''

# Method to use for classification
one_class_artefakt_detection = False
Markov = False

if one_class_artefakt_detection==False: 
    print "Classifier used is decision tree based"
    print "--------------------------------"
    rfc  = RandomForestClassifier()
    dtr  = DecisionTreeClassifier() 
    gb   = GradientBoostingClassifier(n_estimators=1000)
    ada_dct = AdaBoostRegressor(DecisionTreeClassifier(max_depth=2),n_estimators=600, random_state=np.random.RandomState(1))
    lda = LinearDiscriminantAnalysis();
    # Chose an estimator if it
    # is not one-class-SVM  or HMM
    estimator = rfc
    estimator.fit(X_train,Y_train)

    if Markov:
        markov = hmm.GaussianHMM(n_components=2, n_iter=500, params="mcst", init_params="mcst", covariance_type="full")
        estimator = markov
        estimator.fit(X_train)
    
else:
    print "Classifier used is 1 class SVM "
    print "-------------------------------"
    ocsvm = OneClassSVM(nu=0.05,gamma=0.1)
    estimator = ocsvm
    estimator.fit(X_train[Y_train==1,:]) # Train on artefakt-cleaned part of training data set

    # Additionaly identify outliers
    X_train_outliers = X_train[Y_train==-1,:]
    X_train_healthy  = X_train[Y_train==1,:]
    X_test_outliers  = X_test[Y_test==-1,:]
    X_test_healthy   = X_test[Y_test==1,:]

    clf = estimator
    # Copied code...
    y_pred_train_healthy  = clf.predict(X_train_healthy)
    y_pred_train_outliers = clf.predict(X_train_outliers)
    y_pred_test_healthy   = clf.predict(X_test_healthy)
    y_pred_test_outliers  = clf.predict(X_test_outliers)

    n_error_train_healthy  = y_pred_train_healthy[y_pred_train_healthy == -1].size
    n_error_train_outliers = y_pred_train_outliers[y_pred_train_outliers == 1].size
    n_error_test_healthy   = y_pred_test_healthy[y_pred_test_healthy == -1].size
    n_error_test_outliers  = y_pred_test_outliers[y_pred_test_outliers == 1].size

    xx, yy = np.meshgrid(np.linspace(-15, 15, 500), np.linspace(-15, 15, 500))
    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')

    h1 = plt.scatter(X_train_healthy[:, 0], X_train_healthy[:, 1], c='white')
    h2 = plt.scatter(X_test_healthy[:, 0], X_test_healthy[:, 1], c='green')
    o1 = plt.scatter(X_train_outliers[:, 0], X_train_outliers[:, 1], c='red')
    o2 = plt.scatter(X_test_outliers[:, 0], X_test_outliers[:, 1], c='purple')

    plt.axis('tight')
    plt.xlim((-14, 14))
    plt.ylim((-14, 14))
    plt.legend([a.collections[0], h1, h2, o1, o2],
               ["learned frontier", "healthy training observations",
                "healthy test observations", "abnormal training observations",
                "abnormal test observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train healthy: %d/200 ; errors test healthy: %d/40 ; "
        "errors train outliers: %d/40 ; errors test outliers: %d/40"
        % (n_error_train_healthy, n_error_test_healthy, n_error_train_outliers,n_error_test_outliers))
    plt.show()

# Evaluate Predictions
print "Evaluate artefakt detection on training data set (shape is"+str(np.shape(X_train))+"): "
predictions = estimator.predict(X_train)
if Markov:
    tmp = np.array(predictions)
    if len(tmp[tmp==0])<len(tmp[tmp==1]):
        predictions[tmp==0] = 1
        predictions[tmp==1] = -1
    else:
        predictions[tmp==0] = -1
        predictions[tmp==1] = 1
sleeplib.evaluate_artefakt_detection(predictions,Y_train)
print "Evaluate artefakt detection on testing data set (shape is"+str(np.shape(X_test))+"): "
predictions = estimator.predict(X_test)
if Markov:
    tmp = np.array(predictions)
    if len(tmp[tmp==0])<len(tmp[tmp==1]):
        predictions[tmp==0] = 1
        predictions[tmp==1] = -1
    else:
        predictions[tmp==0] = -1
        predictions[tmp==1] = 1
sleeplib.evaluate_artefakt_detection(predictions,Y_test)