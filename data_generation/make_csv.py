"""
Data Preparation. Based on designed experiment, 
the script stacks features and labels and writes them 
into .csv file which is then used for learning and validation.
.csv files get their names based on the experiment number.
"""
import sys
sys.path.insert(0, '../library')
from edfplus import load_edf 
from features import extract_features
from artlib import remove_artefakts
import numpy as np
from sklearn.cross_validation import train_test_split

# --------------------------------------------------------------------------- #
# ----------------------------- CONSTANTS ----------------------------------- #
# --------------------------------------------------------------------------- #
"""
Do not change unless working 
on a different computer (folders) 
or with different data set (interval)!!!!!
"""
DATA_FOLDER   = '/home/djordje/Desktop/EEGData/'
CSV_FOLDER    = '/home/djordje/Desktop/CSVdata/'
INTERVAL      = 4
"""
These are changed according to
desired experiment...
"""
EXP_NUM = int(sys.argv[1]);
print "The chosen experiment is " + str(EXP_NUM)

# --------------------------------------------------------------------------- #
# --------------------- EXPERIMENTS ----------------------------------------- #
# --------------------------------------------------------------------------- #
split_exp = [None]*1000
training_set_exp = [None]*1000
testing_set_exp = [None]*1000
mapping_exp = [None]*1000
eeg_exchange_training = [None]*1000
eeg_exchange_test = [None]*1000
split_random = [None]*1000
artefakt_removal = [None]*1000
feature_extractor_id = [None]*1000
"""
-----------
Experiment1: Train on Wild Types from GABRA and test on Wild Types from GABRA 
* Assumptions:
    (1) We care only about wild types.
    (2) We care only about EEG1(parietal) and EMG.
    (3) We do not take into account state of the light.
    (4) We do not treat Artefakts specially.
-----------
"""
split_exp[1]          = True
split_random[1]       = True
training_set_exp[1]   = ['EEGDataPool/CM/GABRA/WT/GAB2R']
testing_set_exp[1]    = []
mapping_exp[1] = {
    'w': 0,
    'n': 1,
    'r': 2,
    '1': 0, # Wake artefakt
    '2': 1, # NREM artefakt
    '3': 2, # REM  artefakt
    'a': 6,
    '0': 7,
};
eeg_exchange_training[1] = False;
eeg_exchange_test[1] = False;
artefakt_removal[1] = False; #all the same..
feature_extractor_id[1] = 'fourier'
"""
-----------
Experiment2:  Train on Wild Types from GABRA but test on KO from GABRA 
* Assumptions:
    (1) We care only about EEG1(parietal) and EMG.
    (2) We do not take into account state of the light.
    (3) We do not treat Artefakts specially.
-----------
"""
split_exp[2]          = False
split_random[2]       = False
training_set_exp[2]   = ['EEGDataPool/CM/GABRA/WT/GAB2R']
testing_set_exp[2]    = ['EEGDataPool/CM/GABRA/KO/GAB2R']
mapping_exp[2] = {
    'w': 0,
    'n': 1,
    'r': 2,
    '1': 0, # Wake artefakt
    '2': 1, # NREM artefakt
    '3': 2, # REM  artefakt
    'a': 6,
    '0': 7,
};
eeg_exchange_training[2] = False;
eeg_exchange_test[2] = False;
artefakt_removal[2] = False; #all the same..
feature_extractor_id[2] = 'fourier'
"""
-----------
Experiment3: The one Stefan asked for for 17th of March ->WT on WT
-----------
"""
split_exp[3]          = True
split_random[3]       = False
training_set_exp[3]   = ['EEGDataPool/AS/March17WT/']
testing_set_exp[3]    = []
mapping_exp[3] = {
    'w': 0,
    'n': 1,
    'r': 2,
    '1': 0, # Wake artefakt
    '2': 1, # NREM artefakt
    '3': 2, # REM  artefakt
    'a': 6,
    '0': 7,
};
eeg_exchange_training[3] = True;
eeg_exchange_test[3] = False;
artefakt_removal[3] = False; #all the same..
feature_extractor_id[3] = 'fourier'
"""
-----------
Experiment4: The one Stefan asked for for 17th of March ->WT on KO
-----------
"""
''
split_exp[4]          = False
split_random[4]       = False
training_set_exp[4]   = ['EEGDataPool/AS/March17WT/']
testing_set_exp[4]    = ['EEGDataPool/CM/GABRA/KO/March17KO/']
mapping_exp[4] = {
    'w': 0,
    'n': 1,
    'r': 2,
    '1': 0, # Wake artefakt
    '2': 1, # NREM artefakt
    '3': 2, # REM  artefakt
    'a': 6,
    '0': 7,
};
eeg_exchange_training[4] = True;
eeg_exchange_test[4] = False;
artefakt_removal[4] = False; #all the same..
feature_extractor_id[4] = 'fourier'
"""
-----------
Experiment5: Experiment 4 with NREM and REM Artefakts removed from test and training data set.
-----------  Wake Artefakts are considered as wake states...
             For some reason removing wake artefakts results in much much worse (84% --> 57%) !!!!!! Why????
"""
''
split_exp[5]          = False
split_random[5]       = False
training_set_exp[5]   = ['EEGDataPool/AS/March17WT/']
testing_set_exp[5]    = ['EEGDataPool/CM/GABRA/KO/March17KO/']
mapping_exp[5] = {
    'w': 0,
    'n': 1,
    'r': 2,
    '1': 0, # Wake artefakt
    '2': 4, # NREM artefakt
    '3': 5, # REM  artefakt
    'a': 6,
    '0': 7,
};
eeg_exchange_training[5] = True;
eeg_exchange_test[5]     = False;
artefakt_removal[5]      = True; #In this experiment we remove some artefakts
feature_extractor_id[5] = 'fourier'
"""
-----------
Experiment6: The purpose of this experiment is to test
-----------  how efficiently we can perform artefakt detection.
"""         # This specific experiment shows that data sets can have completely different EMG signals and thus the threshold
            # values differ.
''
split_exp[6]          = False
split_random[6]       = False
training_set_exp[6]   = ['EEGDataPool/CM/GABRA/WT/GAB2R35','EEGDataPool/CM/GABRA/WT/GAB2R36','EEGDataPool/CM/GABRA/WT/GAB2R32']
testing_set_exp[6]    = ['EEGDataPool/CM/GABRA/WT/GAB2R22']#,'EEGDataPool/CM/GABRA/WT/GAB2R26C']
                        # File 'EEGDataPool/CM/GABRA/WT/GAB2R35D' should be left out??? bad???
                        # bad predictions: 'EEGDataPool/CM/GABRA/WT/GAB2R28'
                        #                  'EEGDataPool/CM/GABRA/WT/GAB2R26'
                        #                  'EEGDataPool/CM/GABRA/WT/GAB2R22'

mapping_exp[6] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakts
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
};
eeg_exchange_training[6] = False;
eeg_exchange_test[6]     = False;
artefakt_removal[6]      = False;
feature_extractor_id[6] = 'hybrid'
"""
-----------
Experiment7: The purpose of this experiment is to test
-----------  how efficiently we can perform artefakt detection.
"""         
''
split_exp[7]          = False
split_random[7]       = False
#training_set_exp[7]   = ['EEGDataPool/CM/GABRA/WT/GAB2R22','EEGDataPool/CM/GABRA/WT/GAB2R26','EEGDataPool/CM/GABRA/WT/GAB2R28']
#testing_set_exp[7]    = ['EEGDataPool/AS/March17WT/']
training_set_exp[7]   = ['EEGDataPool/CM/GABRA/WT/GAB2R26']
testing_set_exp[7]    = ['EEGDataPool/CM/GABRA/WT/GAB2R22']


mapping_exp[7] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
};
eeg_exchange_training[7] = False;
eeg_exchange_test[7]     = False;
artefakt_removal[7]      = False;
feature_extractor_id[7] = 'hybrid'
"""
-----------
Experiment8: The purpose of this experiment is to test
-----------  how efficiently we can perform artefakt detection.
             One file of KO for training and one file of KO for testing.
"""         
''
split_exp[8]          = False
split_random[8]       = False
training_set_exp[8]   = ['EEGDataPool/CM/GABRA/KO/March17KO/GAB2R21B']
testing_set_exp[8]    = ['EEGDataPool/CM/GABRA/KO/March17KO/GAB2R21E']
mapping_exp[8] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
};
eeg_exchange_training[8] = False;
eeg_exchange_test[8]     = False;
artefakt_removal[8]      = False;
feature_extractor_id[8] = 'hybrid'
"""
-----------
Experiment10: Deep Learning artefakt rejection!
-----------  
"""         
''
split_exp[10]          = False
split_random[10]       = False
training_set_exp[10]   = ['EEGDataPool/CM/GABRA/WT/GAB2R26']
testing_set_exp[10]    = ['EEGDataPool/CM/GABRA/WT/GAB2R22']


mapping_exp[10] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
};
eeg_exchange_training[10] = False;
eeg_exchange_test[10]     = False;
artefakt_removal[10]      = False;
feature_extractor_id[10] = 'imaging'
"""
-----------
Experiment11: Deep Learning artefakt rejection!
-----------  
"""         
''
split_exp[11]          = False
split_random[11]       = False
training_set_exp[11]   = ['EEGDataPool/CM/GABRA/WT/GAB2R26']
testing_set_exp[11]    = ['EEGDataPool/CM/GABRA/WT/GAB2R22']

mapping_exp[11] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
};
eeg_exchange_training[11] = False;
eeg_exchange_test[11]     = False;
artefakt_removal[11]      = False;
feature_extractor_id[11] = 'raw'
"""
-----------
Experiment12: Deep Learning artefakt rejection!
-----------  
"""         
''
split_exp[12]          = False
split_random[12]       = False
training_set_exp[12]   = ['EEGDataPool/CM/GABRA/WT/GAB2R28']
testing_set_exp[12]    = ['EEGDataPool/CM/GABRA/WT/GAB2R26']


mapping_exp[12] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
};
eeg_exchange_training[12] = False;
eeg_exchange_test[12]     = False;
artefakt_removal[12]      = False;
feature_extractor_id[12] = 'imaging'
"""
-----------
Experiment13: Deep Learning artefakt rejection!
-----------  
"""         
''
split_exp[13]          = True
split_random[13]       = True
training_set_exp[13]   = ['EEGDataPool/CM/GABRA/WT/GAB2R22','EEGDataPool/CM/GABRA/WT/GAB2R26','EEGDataPool/CM/GABRA/WT/GAB2R28']
testing_set_exp[13]    = []


mapping_exp[13] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
};
eeg_exchange_training[13] = False;
eeg_exchange_test[13]     = False;
artefakt_removal[13]      = False;
feature_extractor_id[13] = 'imaging'

"""
-----------
Experiment1: Double Scored Wild Type data:
-----------  -----------------------------  
"""                                        # random split and shuffling
                                           # only one channel used (EEG1)
                                           # EEG1-frontal, EEG2-parietal
''
split_exp[1]          = True
split_random[1]       = True
training_set_exp[1]   = ['DoubleScored/WildTypes/Intersection/']
testing_set_exp[1]    = []


mapping_exp[1] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
    'a': -1, # ????
};
eeg_exchange_training[1] = False;
eeg_exchange_test[1]     = False;
artefakt_removal[1]      = False;
feature_extractor_id[1] = 'imaging'

"""
-----------
Experiment2: Double Scored Wild Type data:
-----------  -----------------------------  
"""                                        # random split and shuffling
                                           # All channels used !!!
                                           # EEG1-frontal, EEG2-parietal
''
split_exp[2]          = True
split_random[2]       = True
training_set_exp[2]   = ['DoubleScored/WildTypes/Intersection/']
testing_set_exp[2]    = []


mapping_exp[2] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
    'a': -1, # ????
};
eeg_exchange_training[2] = False;
eeg_exchange_test[2]     = False;
artefakt_removal[2]      = False;
feature_extractor_id[2] = 'full_imaging'

"""
-----------
Experiment3: Double Scored Wild Type data:
-----------  -----------------------------  
"""                                        # random split and shuffling
                                           # All channels used !!!
                                           # EEG1-frontal, EEG2-parietal
                                           # Instead of working with temporal signals, 
                                           # here we try to work with FOURIER signals
''
split_exp[3]          = True
split_random[3]       = True
training_set_exp[3]   = ['DoubleScored/WildTypes/Intersection/']
testing_set_exp[3]    = []


mapping_exp[3] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
    'a': -1, # ????
};
eeg_exchange_training[3] = False;
eeg_exchange_test[3]     = False;
artefakt_removal[3]      = False;
feature_extractor_id[3] = 'full_fourier'

"""
-----------
Experiment4: Double Scored Mutant data:
-----------  -----------------------------  
"""                                        # random split and shuffling
                                           # All channels used !!!
                                           # EEG1-frontal, EEG2-parietal
''
split_exp[4]          = True
split_random[4]       = True
training_set_exp[4]   = ['DoubleScored/Mutants/Intersection/']#['DoubleScored/Mutants/Intersection/AS52C']
testing_set_exp[4]    = []


mapping_exp[4] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
    'a': -1, # ????
    "'": -1, # ????
};
eeg_exchange_training[4] = False;
eeg_exchange_test[4]     = False;
artefakt_removal[4]      = False;
feature_extractor_id[4] = 'full_imaging'


"""
-----------
Experiment5: Double Scored Wild Type data:
-----------  -----------------------------  
"""                                        # EXPERIMENT FOR GENERATING ERROR STATISTICS
                                           # random split and shuffling, but we maintain the indices of samples
                                           # All channels used !!!
                                           # EEG1-frontal, EEG2-parietal
''
split_exp[5]          = True
split_random[5]       = True
training_set_exp[5]   = ['DoubleScored/WildTypes/Intersection/']
testing_set_exp[5]    = []


mapping_exp[5] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
    'a': -1, # ????
    "'": -1,
    '4': -1
};
eeg_exchange_training[5] = False;
eeg_exchange_test[5]     = False;
artefakt_removal[5]      = False;
feature_extractor_id[5] = 'full_imaging'

"""
-----------
Experiment6: Double Scored Mutant data:
-----------  -----------------------------  
"""                                        # EXPERIMENT FOR GENERATING ERROR STATISTICS
                                           # random split and shuffling, but we maintain the indices of samples
                                           # All channels used !!!
                                           # EEG1-frontal, EEG2-parietal
''
split_exp[6]          = True
split_random[6]       = True
training_set_exp[6]   = ['DoubleScored/Mutants/Intersection/']
testing_set_exp[6]    = []


mapping_exp[6] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
    'a': -1, # ????
    "'": -1,
    '4': -1
};
eeg_exchange_training[6] = False;
eeg_exchange_test[6]     = False;
artefakt_removal[6]      = False;
feature_extractor_id[6] = 'full_imaging'

"""
-----------
Experiment7: Double Scored Mutant + WildType data:
-----------  -------------------------------------
"""                                        

split_exp[7]          = True
split_random[7]       = True
training_set_exp[7]   = ['DoubleScored/WildTypes/Intersection/','DoubleScored/Mutants/Intersection/']
testing_set_exp[7]    = []


mapping_exp[7] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
    'a': -1, # ????
    "'": -1,
    '4': -1
};
eeg_exchange_training[7] = False;
eeg_exchange_test[7]     = False;
artefakt_removal[7]      = False;
feature_extractor_id[7] = 'full_imaging'

"""
-----------
Experiment8: Double Scored WildType to train ====> Double Scored Mutants to test
-----------  --------------------------------------------------------------------
"""                                        

split_exp[8]          = False
split_random[8]       = False
training_set_exp[8]   = ['DoubleScored/WildTypes/Intersection/']
testing_set_exp[8]    = ['DoubleScored/Mutants/Intersection/']


mapping_exp[8] = {
    'w': +1,
    'n': +1,
    'r': +1,
    '1': -1, # Wake artefakt
    '2': -1, # NREM artefakt
    '3': -1, # REM  artefakt
    'a': -1, # ????
    "'": -1,
    '4': -1
};
eeg_exchange_training[8] = False;
eeg_exchange_test[8]     = False;
artefakt_removal[8]      = False;
feature_extractor_id[8] = 'full_imaging'


# --------------------------------------------------------------------------- #
# ------ Parametrize script based on the selected experiment ---------------- #
# ----------------- and extract features and labels ------------------------- #
# --------------------------------------------------------------------------- #
training_set          = training_set_exp[EXP_NUM]
testing_set           = testing_set_exp[EXP_NUM]
mapping               = mapping_exp[EXP_NUM]
split                 = split_exp[EXP_NUM]
split_random          = split_random[EXP_NUM]
eeg_exchange_training = eeg_exchange_training[EXP_NUM]
eeg_exchange_test     = eeg_exchange_test[EXP_NUM]
artefakt_removal      = artefakt_removal[EXP_NUM]
feature_extractor_id  = feature_extractor_id[EXP_NUM]

if split:
    '''
    In case we do not have separate testing data set
    we allocate a part of training data for testing purposes
    '''
    print 'Get training data set'
    [features,labels] = extract_features(feature_extractor_id,DATA_FOLDER,training_set,mapping,INTERVAL,eeg_exchange_training)
    # Remove artefakts if requested by the experiment
    if artefakt_removal:
        print 'Remove artefakts from training data set'
        [features,labels] = remove_artefakts(features,labels)

    print 'Split training data set for testing'
    if split_random==False:
        print "non-random split"
        tra_size = len(labels)-len(labels)/4
        Xt = features[0:tra_size]
        Xv = features[tra_size:len(labels)]
        Yt = labels[0:tra_size]
        Yv = labels[tra_size:len(labels)]
    else:
        print "random split"
        Xt, Xv, Yt, Yv = train_test_split(features, labels, test_size=0.25, random_state=42)

else:
    '''
    In case we have a separate
    testing and training data set
    '''
    print 'Get training data set'
    [Xt,Yt] = extract_features(feature_extractor_id,DATA_FOLDER,training_set,mapping,INTERVAL,eeg_exchange_training)
    print 'Get testing data set'
    [Xv,Yv] = extract_features(feature_extractor_id,DATA_FOLDER,testing_set,mapping,INTERVAL,eeg_exchange_test)
    # Remove artefakts if requested by the experiment
    if artefakt_removal:
        print 'Remove artefakts from both data sets'
        [Xt,Yt] = remove_artefakts(Xt,Yt)
        [Xv,Yv] = remove_artefakts(Xv,Yv)

'''
Special!
'''
# PAY ATTENTION !!! Allow only one column fo training data
Yt = Yt[:,0]

# Adding the id-column to the data
idx_Xt = np.linspace(1, len(Xt), len(Xt)).astype(np.int);
idx_Xv = np.linspace(1 + len(Xt), len(Xt) + len(Xv), len(Xv)).astype(np.int);

# Concatenate
Xt = np.c_[idx_Xt, Xt];
Xv = np.c_[idx_Xv, Xv];

# Save data
np.savetxt(CSV_FOLDER+'train_exp'+str(EXP_NUM)+'.csv', np.c_[Xt, Yt.astype(np.int)], fmt='%f', delimiter=',')
np.savetxt(CSV_FOLDER+'test_exp' +str(EXP_NUM)+'.csv', np.c_[Xv, Yv.astype(np.int)], fmt='%f', delimiter=',')
