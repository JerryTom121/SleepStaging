"""
Script used for testing the artefakt identification.
Calls functions from 'artlib' which try to detect artefakts.
"""
import sys; sys.path.insert(0, '../library')
from edfplus import load_edf 
import artlib
import sleeplib
import features
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# sklearn stuff
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC,OneClassSVM
from sklearn import preprocessing

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
# --------------------------------------------------------------------------- #
# --- Read the file and try to detect artefakts in an unsupervised manner --- #
# --------------------------------------------------------------------------- #
# Current mini data set we are testing
#edf_file = "/home/djordje/Desktop/EEGData/EEGDataPool/CM/GABRA/KO/March17KO/GAB2R24E.edf"
#std_file = "/home/djordje/Desktop/EEGData/EEGDataPool/CM/GABRA/KO/March17KO/GAB2R24E.STD"
edf_file = "/home/djordje/Desktop/EEGData/EEGDataPool/CM/GABRA/WT/GAB2R35D.edf"
std_file = "/home/djordje/Desktop/EEGData/EEGDataPool/CM/GABRA/WT/GAB2R35D.STD"

# Some vars
interval_size = 4;
data              = load_edf(edf_file)
sample_rate       = data.sample_rate
samples_per_epoch = interval_size * sample_rate

# Extract labels
raw_labels  = np.genfromtxt(std_file, skip_header=0, dtype=str, comments="4s")
epochs = np.shape(raw_labels)[0]
labels = np.zeros(epochs)
for i in range(int(epochs)):
    labels[i] = mapping[raw_labels[i, 1]]

# Extract EEG and EMG traces
eeg1 = data.X[data.chan_lab.index('EEG1')]
eeg2 = data.X[data.chan_lab.index('EEG2')]
emg  = data.X[data.chan_lab.index('EMG')]
# --------------------------------------------------------------------------- #
# ---------------------------- Debug Output --------------------------------- #
# --------------------------------------------------------------------------- #
# Get relative frequencies
rel_frequencies = artlib.stage_distribution(pd.DataFrame(labels),STAGES)
print "There is " + str(epochs) + " intervals in this file"
print "Occurences of each stage in current training data set: "
print "WAKE = " + str(rel_frequencies[WAKE])
print "NREM = " + str(rel_frequencies[NREM])
print "REM  = " + str(rel_frequencies[REM])
print "WART = " + str(rel_frequencies[WART])
print "NART = " + str(rel_frequencies[NART])
print "RART = " + str(rel_frequencies[RART])
# --------------------------------------------------------------------------- #
# ------------------------- Artefakt Detection ------------------------------ #
# --------------------------------------------------------------------------- #
# For Learning Purposes
X = [] # Features
Y = [] # Labels

hit  = np.zeros(3);
fp   = np.zeros(3);
fn   = np.zeros(3);
eeg1_means = np.zeros(6);
eeg1_devs  = np.zeros(6);
eeg1_maxs   = np.zeros(6);
eeg1_mins   = np.zeros(6);
eeg1_slope = np.zeros(6);

eeg2_means = np.zeros(6);
eeg2_devs  = np.zeros(6);
eeg2_maxs   = np.zeros(6);
eeg2_mins   = np.zeros(6);
eeg2_slope = np.zeros(6);

emg_means  = np.zeros(6);
emg_devs   = np.zeros(6);
emg_maxs    = np.zeros(6);
emg_mins    = np.zeros(6);
emg_slope  = np.zeros(6);

counts     = np.zeros(6);

tmp = 50

for i in range(int(epochs)):
    '''
    In the first round 
    calculate the statistics !!!
    '''
    # Extract Data
    signal1 = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
    signal2 = eeg2[samples_per_epoch * i: samples_per_epoch * (i + 1)]
    signal3 = emg[samples_per_epoch * i: samples_per_epoch * (i + 1)]
    stage = labels[i]

    # Update counters
    counts[stage] = counts[stage] + 1
    eeg1_means[stage] = eeg1_means[stage] + np.mean(signal1)
    eeg1_devs[stage]  = eeg1_devs[stage] + np.std(signal1)
    eeg1_maxs[stage]   = eeg1_maxs[stage] + np.amax(signal1)
    eeg1_mins[stage]   = eeg1_mins[stage] + np.amin(signal1)
    eeg1_slope[stage] = eeg1_slope[stage] + abs(np.mean(signal1[1:tmp])-np.mean(signal1[512-tmp:512]))

    eeg2_means[stage] = eeg2_means[stage] + np.mean(signal2)
    eeg2_devs[stage]  = eeg2_devs[stage] + np.std(signal2)
    eeg2_maxs[stage]   = eeg2_maxs[stage] + np.amax(signal2)
    eeg2_mins[stage]   = eeg2_mins[stage] + np.amin(signal2)
    eeg2_slope[stage] = eeg2_slope[stage] + abs(np.mean(signal2[1:tmp])-np.mean(signal2[512-tmp:512]))

    emg_means[stage]  = emg_means[stage] + np.mean(signal3)
    emg_devs[stage]   = emg_devs[stage] + np.std(signal3)
    emg_maxs[stage]    = emg_maxs[stage] + np.amax(signal3)
    emg_mins[stage]    = emg_mins[stage] + np.amin(signal3)
    emg_slope[stage]  = emg_slope[stage] + abs(np.mean(signal3[1:tmp])-np.mean(signal3[512-tmp:512]))    

# Relevant Statistics
eeg1_means = eeg1_means/counts
eeg1_devs  = eeg1_devs/counts
eeg1_maxs  = eeg1_maxs/counts
eeg1_mins  = eeg1_mins/counts
eeg1_slope = eeg1_slope/counts
eeg2_means = eeg2_means/counts;
eeg2_devs  = eeg2_devs/counts;
eeg2_maxs   = eeg2_maxs/counts;
eeg2_mins   = eeg2_mins/counts;
eeg2_slope = eeg2_slope/counts
emg_means  = emg_means/counts;
emg_devs   = emg_devs/counts;
emg_maxs    = emg_maxs/counts;
emg_mins    = emg_mins/counts;
emg_slope  = emg_slope/counts

# We go 4s by 4s and we try to identify if the current period is an artefakt
for i in range(int(epochs)):
    '''
    In the second round we perform artefakt detection
    '''
    # Extract Signals
    signal1 = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
    signal2 = eeg2[samples_per_epoch * i: samples_per_epoch * (i + 1)]
    signal3 = emg[samples_per_epoch * i: samples_per_epoch * (i + 1)]
    stage   = labels[i]
    # Extract Statistical Features
    eeg1_mean  = np.mean(signal1)
    eeg2_mean  = np.mean(signal2)
    eeg1_entry = np.mean(signal1[1:tmp])
    eeg1_exit  = np.mean(signal1[512-tmp:512])
    eeg2_entry = np.mean(signal2[1:tmp])
    eeg2_exit  = np.mean(signal2[512-tmp:512])
    eeg1_max   = np.max(signal1)
    eeg1_min   = np.min(signal1)
    eeg2_max   = np.max(signal2)
    eeg2_min   = np.min(signal2)
    eeg1_dev   = np.std(signal1)
    eeg2_dev   = np.std(signal2)
    eeg1_99p   = np.percentile(signal1,99)
    eeg2_99p   = np.percentile(signal2,99)
    eeg1_01p   = np.percentile(signal1,1)
    eeg2_01p   = np.percentile(signal2,1)
    fts = [eeg1_mean,eeg1_dev,eeg1_max,eeg1_min,eeg1_99p,eeg1_01p,eeg1_entry,eeg1_exit]
    # Other features
    pass
    # Features and Labels
    if stage>2:
        label = 1
    else:
        label = 0
    # Stack Features
    X = np.vstack([X, fts]) if np.shape(X)[0] else fts
    Y.append(label)

    '''
    [Artefakt Types presentation]
    Crossing of 2 eeg signals
    '''
    crossing1 = eeg1_entry>eeg2_entry and eeg1_exit<eeg2_exit
    crossing2 = eeg1_entry<eeg2_entry and eeg1_exit>eeg2_exit
    crossing = crossing1 or crossing2
    '''
    [Artefakt Types presentation]
    Random high peaks: amplitudes which are very far off from the mean value
    '''
    random_peak_eeg1 = eeg1_max-eeg1_mean>600 or abs(eeg1_min-eeg1_mean)>600 # Effective!!!
    random_peak_eeg2 = eeg2_max-eeg2_mean>260 or abs(eeg2_min-eeg2_mean)>260 
    random_peak = random_peak_eeg1 or random_peak_eeg2
    '''
    [Artefakt Types presentation]
    Crazy EEG1 and peacfull EEG1
    '''
    crazy_eeg1 = (eeg1_max-eeg1_min)>950 # Effective!!! (Similar to the previous one??? -> yes basically...)
    peaceful_eeg2 = eeg1_dev <60 #doesnt work -> use only 'crazy_eeg1' instead
    '''
    [Artefakt Types presentation]
    EEG traces out of the window: I interpret this as EEG period mean deviates from the global mean
    '''
    out_of_window_eeg1 = abs(eeg1_mean)>100 # Not bad and different
    out_of_window_eeg2 = abs(eeg2_mean)>70  # Not bad and different
    out_of_window_eeg  = out_of_window_eeg1 or out_of_window_eeg2
    '''
    [Artefakt Types presentation]
    EEG2 broken: amplitudes go crazy?
    '''
    crazy_eeg2 = (eeg2_min-eeg2_min)>475 # Works but not that good, similar to crazy_eeg1 with lower performance
    '''
    [Artefakt Types presentation]
    EEG1 broken: amplitude very close to 0
    '''
    zero_eeg1 = abs(eeg1_min)<4 #and np.std(signal1)<40 # Doesn work
    '''
    [Artefakt Types presentation]
    EMG movement artefakt: almost constant EMG
    '''
    zero_emg = np.std(signal3)<7 # Doesnt work that well...
    '''
    [Artefakt Detection Poster]
    EEG1 Zero Crossing 
    '''
    eeg1_zero_crossing = (eeg1_entry>0 and eeg1_exit<0) or (eeg1_entry<0 and eeg1_exit>0) # Too many false positives....
    '''
    [Artefakt Detection Poster]
    EEG Slope Thresholding, can be used for detecting high jumps
    '''
    eeg1_steep_slope = abs(eeg1_entry-eeg1_exit)>40 # Not bad but still results overlap with high amplitude conditions...
    eeg2_steep_slope = abs(eeg2_entry-eeg2_exit)>50
    '''
    [Artefakt Types presentation]
    Random peaks revisited: percentile based 
    '''
    # Similar to the previous ones but still gives good results 
    factor = 2
    rp_revisited_eeg1 = eeg1_max>factor*eeg1_99p or eeg1_min<factor*eeg1_01p
    rp_revisited_eeg2 = eeg2_max>factor*eeg2_99p or eeg2_min<factor*eeg2_01p
    rp_revisited_eeg  = rp_revisited_eeg1 or rp_revisited_eeg2

    '''
    Perform Artefakt Detection
    '''
    #artefakt_condition = crazy_eeg1 or out_of_window_eeg
    artefakt_condition = crazy_eeg1
    # Detect artefakt according to rules specified above
    if artefakt_condition:
        # If we have false positive
        if stage <= 2:
            fp[stage-3] = fp[stage-3]+1

            '''
            if i==8956:

                plt.plot(signal1)
                plt.plot(signal2)
                plt.show()

                print "\n\nStage"
                print stage
                print "i = "+str(i)
                print "Means"
                print np.mean(signal1)
                print np.mean(signal2)
                print np.mean(signal3)
                print "Devs"
                print np.std(signal1)
                print np.std(signal2)
                print np.std(signal3)
                print "Max"
                print np.max(signal1)
                print np.max(signal2)
                print np.max(signal3)
                print "Min"
                print np.min(signal1)
                print np.min(signal2)
                print np.min(signal3)
            '''
        # otherwise..
        else:
            hit[stage-3] = hit[stage-3]+1
    else:
        # if we have false negative
        if stage > 2:
            fn[stage-3] = fn[stage-3]+1
'''
            if i==10460:
                plt.plot(signal1)
                plt.show()

                print np.amax(signal1)
                print np.percentile(signal1,percent)
                print np.amin(signal1)
                print np.percentile(signal1,100-percent)

                print "\n\nStage"
                print stage
                print "i = "+str(i)
                print "Means"
                print np.mean(signal1)
                print np.mean(signal2)
                print np.mean(signal3)
                print "Devs"
                print np.std(signal1)
                print np.std(signal2)
                print np.std(signal3)
                print "Max"
                print np.max(signal1)
                print np.max(signal2)
                print np.max(signal3)
                print "Min"
                print np.min(signal1)
                print np.min(signal2)
                print np.min(signal3)
'''



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
# Scaling
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train.astype(np.float))
X_test  = scaler.transform(X_test.astype(np.float))



# Method to use for classification
one_class_artefakt_detection = True


if one_class_artefakt_detection==False:
    print "Classifier used is decision tree based"
    print "--------------------------------"
    rfc  = RandomForestClassifier()
    dtr  = DecisionTreeClassifier() 
    gb   = GradientBoostingClassifier(n_estimators=1000)
    ada_dct = AdaBoostRegressor(DecisionTreeClassifier(max_depth=2),n_estimators=600, random_state=np.random.RandomState(1))
    estimator = rfc
    estimator.fit(X_train,Y_train)
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

    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
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
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
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


print "Evaluate artefakt detection on training data set (shape is"+str(np.shape(X_train))+"): "
sleeplib.evaluate_artefakt_detection(estimator,X_train,Y_train)
print "Evaluate artefakt detection on testing data set (shape is"+str(np.shape(X_test))+"): "
sleeplib.evaluate_artefakt_detection(estimator,X_test,Y_test)


#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------#
'''
print "\nEEG1 mean of mean,std,max,min,max-min,slope\n"
print eeg1_means
print eeg1_devs
print eeg1_maxs
print eeg1_mins
print eeg1_maxs-eeg1_mins
print eeg1_slope

print "\nEEG2 mean of mean,std,max,min,max-min,slope\n"
print eeg2_means
print eeg2_devs
print eeg2_maxs
print eeg2_mins
print eeg2_maxs-eeg2_mins
print eeg2_slope
print "\nEMG mean of mean,std,max,min,max-min,slope\n"
print emg_means
print emg_devs
print emg_maxs
print emg_mins
print emg_maxs-emg_mins
print emg_slope


print "----------------------------------------"
print "Hits:            " + str(hit)
print "False Positives: " + str(fp)
print "False Negatives: " + str(fn)
print "----------------------------------------"
'''