from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
import math

from scipy import signal

POLY_DEGREE = 2


# Read Data
# #########
def readData(csv_folder,exp_num,num_features):
    # Input data
    train_file = csv_folder+'train_exp'+str(exp_num)+'.csv'
    test_file  = csv_folder+'test_exp'+str(exp_num)+'.csv'
    # Read training data -> Features are enumarated from 1 to N
    train_input  = pd.read_csv(train_file, header=-1)
    train_output = train_input[num_features+1].to_frame('label')
    del train_input[0]
    del train_input[num_features+1]
    # Read test data -> Features are enumerated from 1 to N
    test_input = pd.read_csv(test_file, header=-1)
    test_ids = test_input[0]
    del test_input[0]
    return [train_input,train_output,test_input,test_ids]


def read_data(csv_folder,exp_num):
    '''Fixed version of 'readData'

    For a given folder to read csv file from and for a given experiment,
    returns corresponding training and testing data sets.

    csv_folder: path to folder which contains .csv files.

    exp_num: number of experiment used to create a data set.
    '''

    # Read from files
    training_file           = np.genfromtxt(csv_folder+'train_exp'+str(exp_num)+'.csv', delimiter=',')
    testing_file            = np.genfromtxt(csv_folder+'test_exp'+str(exp_num)+'.csv', delimiter=',')
    #testing_file_no_labels  = np.genfromtxt(csv_folder+'test_exp'+str(exp_num)+'.csv', delimiter=',')
    #testing_file_labels     = np.genfromtxt(csv_folder+'solution_exp'+str(exp_num)+'.csv', delimiter=',', skip_header=1)

    # Distinguish features and labels
    X_train = training_file[:,1:-1]
    Y_train = training_file[:,-1]
    X_test  = testing_file[:,1:-1]
    Y_test  = testing_file[:,-1]
    #X_test  = testing_file_no_labels[:,1:]
    #Y_test  = testing_file_labels[:,-1]

    # Return values
    return [X_train,Y_train,X_test,Y_test]

# Augment Feature Space
# #####################
def is1(x):
	return 1/(x+1)
def il1(x):
	return 1/(np.log1p(x)*np.log1p(x)+1)
def augmentFSpace(data):
    # New Fetures derived experimentally
  #  data_rec   = FunctionTransformer(np.reciprocal).transform(data)#data.applymap(lambda x: 1/x)-#data_lg1   = FunctionTransformer(np.log1p).transform(data)
    #data_is1   = FunctionTransformer(is1).transform(data)   # GOOD ONE!!!
    data_il1   = FunctionTransformer(il1).transform(data)
    #data_new   = data.applymap(lambda x: np.log(x)/x)
    #data_sqr   = data.applymap(lambda x: x*x)               # REALLY GOOD ONE!!!!!
    #data_sig   = data.applymap(lambda x: 1/(1+math.exp(x)))     # OK ONE!
    #data_exp   = data.applymap(lambda x: math.exp(x))           # GOOD ONE!!!
  #  data = pd.DataFrame(np.concatenate((data.as_matrix(),data_rec), axis=1))
    #data = pd.DataFrame(np.concatenate((data.as_matrix(),data_lg1), axis=1))
    #data = pd.DataFrame(np.concatenate((data.as_matrix(),data_is1), axis=1))
    data = pd.DataFrame(np.concatenate((data.as_matrix(),data_il1), axis=1))
    #data = pd.DataFrame(np.concatenate((data.as_matrix(),data_new.as_matrix()), axis=1))
    #data = pd.DataFrame(np.concatenate((data.as_matrix(),data_sqr.as_matrix()), axis=1))
    #data = pd.DataFrame(np.concatenate((data.as_matrix(),data_sig.as_matrix()), axis=1))
    #data = pd.DataFrame(np.concatenate((data.as_matrix(),data_exp.as_matrix()), axis=1))
    # Add polynomial features to our feature space
    #
    #poly = preprocessing.PolynomialFeatures(POLY_DEGREE).fit(data)
    #data = pd.DataFrame(poly.fit_transform(data))
    
    print("After extension of our feature space we have " + str(len(data.columns)) + " features")    
    return data

# Print Distribution of Sleep Stages
# ##################################
def state_distributions(output,WAKE,NREM,REM):
    '''
    Print out the distribution percentage
    for each sleeping stage 
    '''
    n_states = output.count()[0]
    # wake state
    n_WAKE_states = output.stack().value_counts(dropna=False)[WAKE];
    print "                 WAKE = "+str(n_WAKE_states*1.0/n_states)
    # nrem state
    n_NREM_states = output.stack().value_counts(dropna=False)[NREM];
    print "                 NREM = "+str(n_NREM_states*1.0/n_states)
    # rem state
    try:   
        n_REM_states = output.stack().value_counts(dropna=False)[REM];
        print "                 REM  = "+str(n_REM_states*1.0/n_states)
    except:
        print "                 REM  = 0"
    print "-----------------------------------------------------------------------------"

# Print Distribution of Sleep Stages
# ##################################
def state_transitions(output,WAKE,NREM,REM):
    '''
    Print out the state transtition
    statistics for the given labels
    '''
    NSTATES = 3
    transition_matrix = np.zeros((NSTATES,NSTATES))
    num_labels = len(output)
    # Update Transition Matrix
    for i in range(0,num_labels-1):
        from_label = int(output.loc[i,'label'])
        to_label   = int(output.loc[i+1,'label'])
        transition_matrix[from_label,to_label] = transition_matrix[from_label,to_label]+1
    # Normalize Transition Matrix
    for i in range(0,NSTATES):
        transition_matrix[i] = transition_matrix[i]/np.sum(transition_matrix[i])
    # Output
    print "The transition matrix of the training data is:"
    print transition_matrix
    print "-----------------------------------------------------------------------------"

def evaluate_predictions(predictions,solutions,stages):
    '''
    Validate the quality of predictions,
    by comparing them to the ground truth
    '''
    P = predictions.loc[solutions['Label'].isin(stages)]
    S = solutions.loc[solutions['Label'].isin(stages)]
    C = S['Label'] == P['Label']
    print C.value_counts(normalize=True)
    print "-----------------------------------------------------------------------------"


def include_temporal_features(input,output):

    return [input,output]

def evaluate_artefakt_detection(predictions,Y_test):

    # evaluate
    num_hits = 0
    num_fp = 0
    num_fn = 0

    for i in range(len(predictions)):
        # If not Artefakt
        if predictions[i]==+1:

            if Y_test[i]==-1: # false negative
                num_fn = num_fn + 1
                #print "from evaluate_artefakt_detection, false negative: " + str(i)
        # if Artefakt
        else:
            
            if Y_test[i]==+1: # false positive
                num_fp = num_fp + 1
            else:             # hit
                num_hits = num_hits + 1

    print "Overall samples = " + str(len(predictions))
    print "Number of hits = " + str(num_hits)
    print "Number of fps  = " + str(num_fp)
    print "Number of fns  = " + str(num_fn) 
    print "-----------------------------------"


def bandpass(sig,band,fs):
    B,A = signal.butter(5, np.array(band)/(fs/2), btype='bandpass')
    return signal.lfilter(B, A, sig, axis=0)


def evaluate(predictions,truth):
    # Evaluate quality of predictions
    hits = 0
    miss = 0
    fpos = 0
    for i in range(len(predictions)):

        # Artifact
        if truth[i] == -1:
            # Detected
            if predictions[i] == -1:
                hits = hits + 1
            # Missed
            else:
                miss = miss + 1
        # Non-artifact
        else:
            # False positive
            if predictions[i] == -1:
                fpos = fpos + 1

    # print stats
    print("Hits: "+str(hits))
    print("Fpos: "+str(fpos))
    print("Miss: "+str(miss))