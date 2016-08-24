'''
Part of the eeg library which deals with feature 
and label extraction.
'''
from PIL import Image
import os
import sys
from edfplus import load_edf 
#https://github.com/breuderink/eegtools/blob/master/eegtools/io/edfplus.py
from numpy import fft
import pywt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA,FastICA
from scipy import signal
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import scipy

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def extract_features(feature_extractor_id,data_folder,file_sets,mapping,interval=4,exchange_eeg=False):
    '''Extract features and corresponding labels from the given files.

     This is a general function which extracts and stacks
     features and labels of each file that is part of our data set
     onto a big matrix which represents our data set

     feat_ext: type of the feature extractor to use.
               current possiblities are 'fourier' and 'statistical'

     file_sets: contains acronym of the files to be used. Each file which contains 
                this acronym in the name is included into the data set.

     mapping: defines mapping between raw labels to the labels we want.

     interval: length in seconds of each epoch.

     exchange_eeg: indicator based on which we swap eeg1 and eeg2 or not.
    '''

    # Select feature extractor
    feature_extractors = {
      'temporal': raw_signal_features,
      'spectral': fft_features,
      'hybrid': mixed_features,

      'statistical': statistical_features,
      'hybrid': hybrid_features,
      'imaging': imaging_features,
      'full_fourier': full_fourier_features,
      'raw': raw_features,
    }
    feature_extractor = feature_extractors[feature_extractor_id]
    print 'Feature extractor in use is ' + feature_extractor_id

    # Get features and labels
    features = []
    labels   = []

    for file_set in file_sets:
        # For each file set
        [file_folder,file_id] = str.rsplit(file_set,'/',1)
        for f in os.listdir(data_folder+file_folder):
            if '.edf' in f and file_id in f:
                # For each valid .edf file
                
                # Debug output
                print "--------------------------------------------------"
                print "File "+f+" is now being processed..."

                # Feature extraction
                file_features = feature_extractor(file_path=data_folder+file_folder+'/'+f, interval_size=interval,exchange_eeg=exchange_eeg)
                
                # Label extraction
                num_intervals = np.shape(file_features)[0]
                raw_labels    = np.genfromtxt(data_folder+file_folder+'/'+str.rsplit(f,'.')[0]+'.STD', skip_header=0, dtype=str, comments="4s")
                
                # Number of labels
                nlabels = np.shape(raw_labels)[0]
                print "Number of labels is: " + str(nlabels)

                # Extract labels
                file_labels   = raw_labels[:,1:].copy()
                for i in range(nlabels):
                        file_labels[i,0] = mapping[raw_labels[i,1]]

                # Print out the number of artifacts
                tmp = file_labels[:,0]
                print "Number of artifacts is: " + str(np.count_nonzero(tmp[tmp=='-1']))
                
                # Debug
                # print file_labels

                # Accumulating features and labels
                features = np.vstack([features, file_features]) if np.shape(features)[0] else file_features
                labels   = np.vstack([labels, file_labels])     if np.shape(labels)[0]   else file_labels
                print "The shape of features is: " + str(np.shape(features))
                print "The shape of labels is: " + str(np.shape(labels))

                # Debug output
                print "File "+f+" processed and added to the dataset."
                print "--------------------------------------------------"

    return [features,labels]


# ----------------------------------------------------------------------------------------------- #
# ------------------------------ FEATURE -------------------------------------------------------- #
# ------------------------------ EXTRACTORS ----------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def raw_signal_features(file_path,interval_size=4,exchange_eeg=False, ds_factor=4):
    '''
    Function which extracts imaging features from a given file.
    The extracted features should be used as an input of convolutional network.

    file_path:     Path to the .edf file
    interval_size: How many seconds does each epoch last
    exchange_eeg:  Swap EEG1 and EEG2 signal
    '''
    # Get Raw Data and Sampling Characteristics
    data              = load_edf(file_path)
    sample_rate       = data.sample_rate
    samples_per_epoch = interval_size * sample_rate
    eeg1 = data.X[data.chan_lab.index('EEG1')]
    eeg2 = data.X[data.chan_lab.index('EEG2')]
    emg  = data.X[data.chan_lab.index('EMG')]
    total_size = len(eeg1)
    epochs = total_size / samples_per_epoch


    # Downsampling to reduce signal lenghts
    eeg1 = scipy.signal.decimate(eeg1,ds_factor)
    eeg2 = scipy.signal.decimate(eeg2,ds_factor)
    emg  = scipy.signal.decimate(emg,ds_factor)
    samples_per_epoch = samples_per_epoch/ds_factor

    # Standardize EEG/EMG signals. This is very important and so far I have a very simple 
    # standardization method. I should find one which does not depend on the sleep state distribution, 
    # or number of artefakts in samples.
    #
    # one more thing, we do the standardization per file here, is that good????
    #
    scaler = preprocessing.RobustScaler()
    eeg1   = (scaler.fit_transform(eeg1.reshape(-1,1))).reshape(-1,)
    eeg2   = (scaler.fit_transform(eeg2.reshape(-1,1))).reshape(-1,)
    emg    = (scaler.fit_transform(emg.reshape(-1,1))).reshape(-1,)
   
    # Feature matrix
    X = []
    for i in range(int(epochs)):
        
        # Get signals of current epoch
        eeg1_epoch = eeg1[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        eeg2_epoch = eeg2[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        emg_epoch  = emg[int(samples_per_epoch)  * i: int(samples_per_epoch) * (i + 1)]

        features = np.stack( (eeg1_epoch,eeg2_epoch,emg_epoch), 0).flatten();

        X = np.vstack([X, features]) if np.shape(X)[0] else features

    return X


def fft_features(file_path,interval_size=4,exchange_eeg=False):
    # --------------------------------------------------------------------------- #
    # ---------------- Read .edf file and extract time-series ------------------- #
    # --------------------------------------------------------------------------- #
    data              = load_edf(file_path)
    sample_rate       = data.sample_rate
    samples_per_epoch = interval_size * sample_rate
    # The EEG got recorded with two traces. From the provided readme
    #   > For now we only consider the first EEG trace
    eeg1 = data.X[data.chan_lab.index('EEG1')]
    eeg2 = data.X[data.chan_lab.index('EEG2')]
    emg  = data.X[data.chan_lab.index('EMG')]
    # Perform the exchange of eeg
    # if necessary
    if exchange_eeg:
        tmp = eeg1
        eeg1 = eeg2
        eeg2 = eeg1
        print "exchange done...."
    # Extract information about the number of 
    # epochs for chosen interval time
    total_size = len(eeg1)
    epochs = total_size / samples_per_epoch
    # --------------------------------------------------------------------------- #
    # ---------------- Create array to represent frequency spectrum ------------- #
    # --------------------------------------------------------------------------- #
    # Compute the frequencies corresponding to the eeg_pos_spec indices. Because we
    # sample an entire interval of 'interval' waves at once, need to normalize
    # the frequencies by that factor as well.
    # Due to simetry we calculate only positive frequency spectrum, hence n/2+1
    eeg_pos_spectrum_len = int(samples_per_epoch / 2 + 1);
    eeg_f                = np.linspace(0, eeg_pos_spectrum_len - 1, eeg_pos_spectrum_len) / interval_size
    # --------------------------------------------------------------------------- #
    # ---------------- Create frequency buckets --------------------------------- #
    # --------------------------------------------------------------------------- #
    # The indices from the power spectrum array into a bin.
    # For each feature there is a group of indices for the 
    # corresponding frequency bucket
    eeg1_bin_idx = []
    eeg2_bin_idx = []
    # The following uses standard frequency buckets as they are also often used
    # in the sleep literature.
    # EEG1
    eeg1_bin_idx.append(np.where(np.logical_and(0.49 < eeg_f, eeg_f <= 5))[0])    # delta
    eeg1_bin_idx.append(np.where(np.logical_and(5    < eeg_f, eeg_f <= 9))[0])    # theta
    eeg1_bin_idx.append(np.where(np.logical_and(9    < eeg_f, eeg_f <= 15))[0])   # alpha
    eeg1_bin_idx.append(np.where(np.logical_and(15   < eeg_f, eeg_f <= 23))[0])   # ...
    eeg1_bin_idx.append(np.where(np.logical_and(23   < eeg_f, eeg_f <= 32))[0])
    eeg1_bin_idx.append(np.where(np.logical_and(32   < eeg_f, eeg_f <= 64))[0])
    # EEG2
    eeg2_bin_idx.append(np.where(np.logical_and(0.49 < eeg_f, eeg_f <= 5))[0])    # delta
    eeg2_bin_idx.append(np.where(np.logical_and(5    < eeg_f, eeg_f <= 9))[0])    # theta
    eeg2_bin_idx.append(np.where(np.logical_and(9    < eeg_f, eeg_f <= 15))[0])   # alpha
    eeg2_bin_idx.append(np.where(np.logical_and(15   < eeg_f, eeg_f <= 23))[0])   # ...
    eeg2_bin_idx.append(np.where(np.logical_and(23   < eeg_f, eeg_f <= 32))[0])
    eeg2_bin_idx.append(np.where(np.logical_and(32   < eeg_f, eeg_f <= 64))[0])
    # EMG: Filter the frequencies for the EMG between 4 and 40.
    emg_bin = np.where(np.logical_and(4 <= eeg_f, eeg_f <= 40))[0]
    # --------------------------------------------------------------------------- #
    # - Calculate FT on 4s intervals and energy for each bucket within interval - #
    # --------------------------------------------------------------------------- #
    # Array to contain the resulting preprocessed data:
    # rows are 4s intervals and columns are features/bucket energies,
    # the last column is energy of EMG
    features = np.zeros((int(epochs), len(eeg1_bin_idx) + len(eeg2_bin_idx) + 1))
   # artefakt_features = np.zeros((epochs,1))
    # Perform FFT
    for i in range(int(epochs)):

        # Get signals of current epoch
        eeg1_epoch = eeg1[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        eeg2_epoch = eeg2[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        emg_epoch  = emg[int(samples_per_epoch)  * i: int(samples_per_epoch) * (i + 1)]
        # Compute the FFT on the EEG  and EMG data. Using real-fft to get smaller
        # output vector. Converting the fourier coefficients to energies by taking
        # the squared absolute value on the (complex) fourier coefficient.
        eeg1_pos_spectrum = np.abs(fft.rfft(eeg1_epoch)) ** 2
        eeg2_pos_spectrum = np.abs(fft.rfft(eeg2_epoch)) ** 2
        emg_pos_spectrum  = np.abs(fft.rfft(emg_epoch)) ** 2
        # Compute the sums over the frequency buckets of the EEG signal.
        for j in range(len(eeg1_bin_idx)):
            features[i, j] = np.sum(eeg1_pos_spectrum[eeg1_bin_idx[j]])
        #    
        for j in range(len(eeg2_bin_idx)):
            features[i, len(eeg1_bin_idx)+j] = np.sum(eeg2_pos_spectrum[eeg2_bin_idx[j]])    
        # The EMG power signal goes into a single bucket.
        features[i, -1] = np.sum(emg_pos_spectrum[emg_bin])
        # Get The first artefakt feature - EMG average amplitude
       # artefakt_features[i,0] = 

    # Normalize using log transformation.
    # Based on the paper: "Frequency Domain Analysis of Sleep EEG for Visualization and Automated State Detection"
    # > Log scaled components show better statistical properties. As a consequence,
    # > the power spectrogram is first scaled using a log transformation.
    # features = np.log(features);
    scaler = preprocessing.RobustScaler()
    features = scaler.fit_transform(features)

    return features


def mixed_features(file_path,interval_size=4,exchange_eeg=False, ds_factor=4):

    # get fourier features
    fftfeatures = fft_features(file_path,interval_size,exchange_eeg)
    # get raw temporal signal features
    rawfeatures = raw_signal_features(file_path,interval_size,exchange_eeg,ds_factor)
    # stack them into one matrix
    features = np.hstack([rawfeatures,fftfeatures])
    # return
    return features
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

























def statistical_features(file_path,interval_size=4,exchange_eeg=False):
    '''
    Function which extracts statistical features (mean,deviation,max/min values....) from a given file:

    file_path:     Path to the .edf file
    interval_size: How many seconds does each epoch last
    exchange_eeg:  Swap EEG1 and EEG2 signal
    '''

    data              = load_edf(file_path)
    sample_rate       = data.sample_rate
    samples_per_epoch = interval_size * sample_rate

    eeg1 = data.X[data.chan_lab.index('EEG1')]
    eeg2 = data.X[data.chan_lab.index('EEG2')]
    emg  = data.X[data.chan_lab.index('EMG')]

    total_size = len(eeg1)
    epochs = total_size / samples_per_epoch

    # Feature matrix
    X = []
    tmp = 50
    for i in range(int(epochs)):

        eeg1_epoch = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        
        # Extract Statistical Features
        eeg1_mean  = np.mean(eeg1_epoch)
        eeg1_entry = np.mean(eeg1_epoch[1:tmp])
        eeg1_exit  = np.mean(eeg1_epoch[512-tmp:512])
        eeg1_max   = np.max(eeg1_epoch)
        eeg1_min   = np.min(eeg1_epoch)
        eeg1_dev   = np.std(eeg1_epoch)
        eeg1_99p   = np.percentile(eeg1_epoch,99)
        eeg1_95p   = np.percentile(eeg1_epoch,95)
        eeg1_80p   = np.percentile(eeg1_epoch,80)
        eeg1_50p   = np.percentile(eeg1_epoch,50)
        eeg1_05p   = np.percentile(eeg1_epoch,5)
        eeg1_01p   = np.percentile(eeg1_epoch,1)

        features = [eeg1_mean,eeg1_dev,eeg1_50p,eeg1_max,eeg1_min,eeg1_entry,eeg1_exit]
        features = [eeg1_mean,max(eeg1_max,abs(eeg1_min))] # Very good for distinguishing outliers in KO case !!!!!!

        # Stack features
        X = np.vstack([X, features]) if np.shape(X)[0] else features

    return X


def hybrid_features(file_path,interval_size=4,exchange_eeg=False):
    # Useful features found so far:
    #           EMG spectrum          -> ffeatures[:,6]                  } be careful with
    #           EMG max amplitude     -> np.max(np.absolute(emg_epoch))  } these 2   
    #
    #           statistical_features  -> np.max(np.gradient(eeg1_epoch))
    #                                 -> max(np.max(eeg1_epoch),abs(np.min(eeg1_epoch))) } be careful with this as well
    #                                 
    #           slope pay attention   -> slope1*slope2: I VERIFIED IT DETECTS SOME ARTEFAKTS
    #           for hmms is not bad   -> window_max-eeg1_max
    #
    # The Artefakts that I miss come in series !!!!!!!!!!
    #
    #
    
    # The Fourier Features
    ffeatures =  fourier_features(file_path)
    #X = ffeatures[:,[3,6]]
    
    # Get Raw Signals
    data              = load_edf(file_path)
    sample_rate       = data.sample_rate
    print sample_rate
    samples_per_epoch = interval_size * sample_rate
    eeg1 = data.X[data.chan_lab.index('EEG1')]
    eeg2 = data.X[data.chan_lab.index('EEG2')]
    emg  = data.X[data.chan_lab.index('EMG')]
    total_size = len(eeg1)
    epochs = total_size / samples_per_epoch

    # Perform exchange if needed
    if exchange_eeg:
        tmp = eeg1
        eeg1 = eeg2
        eeg2 = eeg1
        print "exchange done...."

    ##
    ## PERFORM ICA
    ## INITIALLY ???!!??
    ##
    '''
    ica = FastICA(n_components=2)

    print np.shape(eeg1)

    signal = np.c_[eeg1,eeg2,emg]
    print np.shape(signal)

    components = ica.fit_transform(signal)

   # plt.plot(components[:,0],c='red')
   # plt.plot(components[:,1],c='green')

    print np.sum(np.absolute(components[:,0]))
    print np.sum(np.absolute(components[:,1]))

    plt.show()
    exit()
    #plt.plot(eeg1_c2,c='blue')
    #plt.plot(s2)
    #plt.plot(s3)
    plt.show()
    '''

    #plt.plot(np.absolute(emg))
    #plt.show()

    # Feature matrix
    X = []
    for i in range(int(epochs)):
        eeg1_epoch = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        eeg2_epoch = eeg2[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        emg_epoch  = emg[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        '''
        # Extract Statistical Features
        eeg1_mean  = np.mean(eeg1_epoch)
        eeg2_mean  = np.mean(eeg2_epoch)
        emg_mean  = np.mean(emg_epoch)

        eeg1_abs_mean = np.mean(np.absolute(eeg1_epoch))
        eeg2_abs_mean = np.mean(np.absolute(eeg2_epoch))
        emg_abs_mean = np.mean(np.absolute(emg_epoch))

        eeg1_95p   = max(np.percentile(eeg1_epoch,95),abs(np.percentile(eeg1_epoch,5)))
        eeg2_95p   = max(np.percentile(eeg2_epoch,95),abs(np.percentile(eeg2_epoch,5)))
        emg_95p   = max(np.percentile(emg_epoch,95),abs(np.percentile(emg_epoch,5)))
        '''

        #eeg1_epoch = (eeg1_epoch-np.mean(eeg1_epoch))/np.std(eeg1_epoch)
        #eeg2_epoch = (eeg2_epoch-np.mean(eeg2_epoch))/np.std(eeg2_epoch)
        #emg_epoch  = (emg_epoch-np.mean(emg_epoch))/np.std(emg_epoch)

        eeg1_max = np.max(np.absolute(eeg1_epoch))
        #eeg2_max = np.max(np.absolute(eeg2_epoch))
        emg_max  = np.max(np.absolute(emg_epoch))

        #eeg1_max_grad = np.max(np.gradient(eeg1_epoch))


        #eeg1_population_range = np.percentile(eeg1_epoch,percentage)-np.percentile(eeg1_epoch,100-percentage)
        #eeg2_population_range = np.percentile(eeg2_epoch,percentage)-np.percentile(eeg2_epoch,100-percentage)
        #emg_population_range  = np.percentile(emg_epoch,percentage)-np.percentile(emg_epoch,100-percentage)

        #eeg1_population_range = np.percentile(np.absolute(eeg1_epoch),percentage)
        #eeg2_population_range = np.percentile(np.absolute(eeg2_epoch),percentage)
        #emg_population_range  = np.percentile(np.absolute(emg_epoch),percentage)

        #percentage = 95
        #eeg1_peak = max(abs(np.max(eeg1_epoch)-np.percentile(eeg1_epoch,95)),abs(np.min(eeg1_epoch)-np.percentile(eeg1_epoch,5)))
        #eeg2_peak = max(abs(np.max(eeg2_epoch)-np.percentile(eeg2_epoch,95)),abs(np.min(eeg2_epoch)-np.percentile(eeg2_epoch,5)))
        #emg_peak = max(abs(np.max(emg_epoch)-np.percentile(emg_epoch,95)),abs(np.min(emg_epoch)-np.percentile(emg_epoch,5)))

        #print str(np.percentile(eeg1_epoch,99)) + " " + str(np.percentile(eeg1_epoch,1))

        #slope1, intercept, r_value, p_value, std_err = linregress(range(512),eeg1_epoch)
        #slope2, intercept, r_value, p_value, std_err = linregress(range(512),eeg2_epoch)
        
        #emg_spec = np.abs(fft.rfft(emg_epoch))**2
        #a = np.sum(emg_spec[50:100])
    
        #emg_diff = np.diff(emg_epoch)
        #emg_diff_2 = np.diff(emg_epoch,n=2)

        #features = [slope1*slope2,eeg1_peak]
        #features = np.hstack([eeg1_epoch,eeg2_epoch])
        #features = eeg1_epoch
        features = [eeg1_max,emg_max]
        #features = [eeg1_peak,emg_peak,eeg2_peak,slope1*slope2,np.max(np.gradient(eeg1_epoch))]
        #features = [np.std(eeg1_epoch),np.std(eeg2_epoch)]
        #print features
        #print np.max(eeg1_epoch-eeg2_epoch)
        #print scipy.stats.kurtosis(eeg1_epoch)
        '''
        window_back = eeg1[samples_per_epoch * max(0,i-4): samples_per_epoch * i+1]
        window_front = eeg1[samples_per_epoch * i: samples_per_epoch * min(int(epochs),i+5)]

        window_front_max = max(np.max(window_front),abs(np.min(window_front)))
        window_back_max = max(np.max(window_back),abs(np.min(window_back)))
        eeg1_max = max(np.max(eeg1_epoch),abs(np.min(eeg1_epoch)))
        '''

        # for hmm
        #features = [window_front_max/window_back_max,max(np.max(emg_epoch),abs(np.min(emg_epoch)))]
        #features = [window_front_max,window_back_max]
        # Stack features
        X = np.vstack([X, features]) if np.shape(X)[0] else features
    #X[:,0] = ffeatures[:,6]
    '''
    X = []
    # Wavelet transform
    for i in range(int(epochs)):
        eeg1_epoch = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        eeg2_epoch = eeg2[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        emg_epoch  = emg[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        cA5, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(eeg1_epoch, 'db4',level=5)
        features = [np.sum(cD4),np.sum(cD5)]
        X = np.vstack([X, features]) if np.shape(X)[0] else features
    '''

    '''
    i = 11512-10800
    s1 = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
    s2 = eeg2[samples_per_epoch * i: samples_per_epoch * (i + 1)]
    s3 = emg[samples_per_epoch * i: samples_per_epoch * (i + 1)]
    eeg1_max = max(np.max(s1),abs(np.min(s1)))
    eeg2_max = max(np.max(s2),abs(np.min(s2)))
    emg_max  = max(np.max(s3),abs(np.min(s3)))
    s1 = s1/eeg1_max
    s2 = s2/eeg2_max
    s3 = s3/emg_max
    print "aaaaa = " + str(scipy.stats.skew(s1))

    plt.plot(s1,c='red')
    plt.plot(np.diff(s1),c='green')
    plt.plot(np.diff(s1,2),c='blue')
    #plt.plot(s2)
    #plt.plot(s3)
    plt.show()
    '''
     

    return X

def imaging_features(file_path,interval_size=4,exchange_eeg=False, ds_factor=4):
    '''
    Function which extracts imaging features from a given file.
    The extracted features should be used as an input of convolutional network.

    file_path:     Path to the .edf file
    interval_size: How many seconds does each epoch last
    exchange_eeg:  Swap EEG1 and EEG2 signal
    '''
    
    # Get Raw Data and Sampling Characteristics
    data              = load_edf(file_path)
    sample_rate       = data.sample_rate
    samples_per_epoch = interval_size * sample_rate
    eeg1 = data.X[data.chan_lab.index('EEG1')]
    eeg2 = data.X[data.chan_lab.index('EEG2')]
    emg  = data.X[data.chan_lab.index('EMG')]
    total_size = len(eeg1)
    epochs = total_size / samples_per_epoch


    # Downsampling to reduce signal lenghts
    eeg1 = scipy.signal.decimate(eeg1,ds_factor)
    eeg2 = scipy.signal.decimate(eeg2,ds_factor)
    emg  = scipy.signal.decimate(emg,ds_factor)
    samples_per_epoch = samples_per_epoch/ds_factor

    # Standardize EEG/EMG signals. This is very important and so far I have a very simple 
    # standardization method. I should find one which does not depend on the sleep state distribution, 
    # or number of artefakts in samples.
    #
    # one more thing, we do the standardization per file here, is that good????
    #
    scaler = preprocessing.RobustScaler()
    eeg1   = (scaler.fit_transform(eeg1.reshape(-1,1))).reshape(-1,)
    eeg2   = (scaler.fit_transform(eeg2.reshape(-1,1))).reshape(-1,)
    emg    = (scaler.fit_transform(emg.reshape(-1,1))).reshape(-1,)
   
    #X_test  = scaler.transform(X_test.astype(np.float))

    # Feature matrix
    X = []
    for i in range(int(epochs)):

        # Get signals of current epoch
        eeg1_epoch = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        eeg2_epoch = eeg2[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        emg_epoch  = emg[samples_per_epoch * i: samples_per_epoch * (i + 1)]

        features = eeg1_epoch

        # Stack new sample on top of the main matrix
        X = np.vstack([X, features]) if np.shape(X)[0] else features

    return X


def full_fourier_features(file_path,interval_size=4,exchange_eeg=False, ds_factor=4):
    '''
    Function which extracts imaging features from a given file.
    The extracted features should be used as an input of convolutional network.

    file_path:     Path to the .edf file
    interval_size: How many seconds does each epoch last
    exchange_eeg:  Swap EEG1 and EEG2 signal
    '''
    
    # Get Raw Data and Sampling Characteristics
    data              = load_edf(file_path)
    sample_rate       = data.sample_rate
    samples_per_epoch = interval_size * sample_rate
    eeg1 = data.X[data.chan_lab.index('EEG1')]
    eeg2 = data.X[data.chan_lab.index('EEG2')]
    emg  = data.X[data.chan_lab.index('EMG')]
    total_size = len(eeg1)
    epochs = total_size / samples_per_epoch

    # Downsampling to reduce signal lenghts
    eeg1 = scipy.signal.decimate(eeg1,ds_factor)
    eeg2 = scipy.signal.decimate(eeg2,ds_factor)
    emg  = scipy.signal.decimate(emg,ds_factor)
    samples_per_epoch = samples_per_epoch/ds_factor

    # Standardize EEG/EMG signals. This is very important and so far I have a very simple 
    # standardization method. I should find one which does not depend on the sleep state distribution, 
    # or number of artefakts in samples.
    #
    # one more thing, we do the standardization per file here, is that good????
    #
    scaler = preprocessing.RobustScaler()
    eeg1   = (scaler.fit_transform(eeg1.reshape(-1,1))).reshape(-1,)
    eeg2   = (scaler.fit_transform(eeg2.reshape(-1,1))).reshape(-1,)
    emg    = (scaler.fit_transform(emg.reshape(-1,1))).reshape(-1,)
   
    # Feature matrix
    X = []
    for i in range(int(epochs)):

        # Get signals of current epoch
        eeg1_epoch = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        eeg2_epoch = eeg2[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        emg_epoch  = emg[samples_per_epoch * i: samples_per_epoch * (i + 1)]

        # Perform fourier transform
        eeg1_epoch = np.abs(fft.rfft(eeg1_epoch))
        eeg2_epoch = np.abs(fft.rfft(eeg2_epoch))
        emg_epoch = np.abs(fft.rfft(emg_epoch))

        '''
        plt.plot(eeg1_epoch)
        plt.plot(eeg2_epoch)
        plt.plot(emg_epoch)
        plt.show()
        '''

        features = np.stack( (eeg1_epoch,eeg2_epoch,emg_epoch), 0).flatten();

        # Stack new sample on top of the main matrix
        X = np.vstack([X, features]) if np.shape(X)[0] else features

    return X


def raw_features(file_path,interval_size=4,exchange_eeg=False, ds_factor=4):
    '''
    Function which extracts raw signal values as features from a given file.
    The extracted features can used as an input of convolutional network or 
    can be further processed.

    file_path:     Path to the .edf file
    interval_size: How many seconds does each epoch last
    exchange_eeg:  Swap EEG1 and EEG2 signal
    '''
    
    # Get Raw Data and Sampling Characteristics
    data              = load_edf(file_path)
    sample_rate       = data.sample_rate
    samples_per_epoch = interval_size * sample_rate
    eeg1 = data.X[data.chan_lab.index('EEG1')]
    eeg2 = data.X[data.chan_lab.index('EEG2')]
    emg  = data.X[data.chan_lab.index('EMG')]
    total_size = len(eeg1)
    epochs = total_size / samples_per_epoch

    # Feature matrix
    X = []
    for i in range(int(epochs)):

        # Get signals of current epoch
        eeg1_epoch = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        eeg2_epoch = eeg2[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        emg_epoch  = emg[samples_per_epoch * i: samples_per_epoch * (i + 1)]

        features = eeg1_epoch

        # Stack new sample on top of the main matrix
        X = np.vstack([X, features]) if np.shape(X)[0] else features

    return X


def spectogram_features():
    pass




def PCA_features(file_path,interval_size=4,exchange_eeg=False):
    '''
    Function which extracts statistical features (mean,deviation,max/min values....) from a given file:

    file_path:     Path to the .edf file
    interval_size: How many seconds does each epoch last
    exchange_eeg:  Swap EEG1 and EEG2 signal
    '''

    data              = load_edf(file_path)
    sample_rate       = data.sample_rate
    samples_per_epoch = interval_size * sample_rate

    eeg1 = data.X[data.chan_lab.index('EEG1')]
    eeg2 = data.X[data.chan_lab.index('EEG2')]
    emg  = data.X[data.chan_lab.index('EMG')]

    total_size = len(eeg1)
    epochs = total_size / samples_per_epoch

    # Feature matrix
    X = []
    tmp = 50
    for i in range(int(epochs)):

        eeg1_epoch = eeg1[samples_per_epoch * i: samples_per_epoch * (i + 1)]
        # Stack features
        X = np.vstack([X, eeg1_epoch]) if np.shape(X)[0] else eeg1_epoch

    pca = PCA(n_components=10)
    pca.fit(X)
    X = pca.transform(X)

    return X

def artefakt_labels(std_file):

    raw_labels  = np.genfromtxt(std_file, skip_header=0, dtype=str, comments="4s")
    
    epochs = np.shape(raw_labels)[0]

    mapping = {                        # Mapping labels to identifiers
        'w': +1,
        'n': +1,
        'r': +1,
        '1': -1, # Wake artefakt
        '2': -1, # NREM artefakt
        '3': -1, # REM  artefakt
    };

    Y = np.zeros(epochs)
    for i in range(int(epochs)):
        Y[i] = mapping[raw_labels[i, 1]]

    return Y