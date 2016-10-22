'''
Part of the eeg library which deals with 
feature and corresponding label extraction.
'''
import os
import sys
from edfplus import load_edf  #https://github.com/breuderink/eegtools/blob/master/eegtools/io/edfplus.py
import numpy as np
from scipy import signal
from sklearn import preprocessing
# speech recognition features
sys.path.append(os.path.abspath('/home/djordje/Software/python_speech_features'))
from python_speech_features import mfcc,logfbank,fbank



def extract_features(Exp,mapping,file_sets):
    '''
    #Description: Extract features and corresponding labels from the given files.
    -----------------------------------------------------------------------------
    @param Exp       => an object which contains the parameters of the experiment
    @param mapping   => shows how to map raw labels into labels which will be used for machine learning
    @param file_sets => path to raw data files from where we extract labels and features
    -----------------------------------------------------------------------------
    @return => stacked matrices of features and labels
    -----------------------------------------------------------------------------
    '''
    # -----------------------------
    # Extract experiment parameters
    # -----------------------------
    feature_extractor_id = Exp.feat_ext
    data_folder          = Exp.eeg_folder
    interval             = Exp.interval
    exchange_eeg         = Exp.extrain
    ds_factor            = Exp.ds_factor
    # ------------------------
    # Select feature extractor
    # ------------------------
    feature_extractors = {
      'temporal':       temporal_raw_features,
      'fourier':        fourier_raw_features,
      'fourier_energy': fourier_energy_features,
    }
    feature_extractor = feature_extractors[feature_extractor_id]
    print '## Feature extractor in use is ' + feature_extractor_id
    print "-------------------------------------"
    # -------------------------------
    # Initialize feature/label matrix
    # -------------------------------
    features = []
    labels   = []

    for file_set in file_sets:
        # For each file set
        [file_folder,file_id] = str.rsplit(file_set,'/',1)
        for f in os.listdir(data_folder+file_folder):
            if '.edf' in f and file_id in f:                # For each valid .edf file             
                # Debug output
                print "File "+f+" is now being processed..."
                # ------------------------------------------------
                # Extract raw signals and sampling characteristics
                # ------------------------------------------------
                data              = load_edf(data_folder+file_folder+'/'+f)
                sample_rate       = data.sample_rate
                samples_per_epoch = interval * sample_rate
                eeg1 = data.X[data.chan_lab.index('EEG1')]
                eeg2 = data.X[data.chan_lab.index('EEG2')]
                emg  = data.X[data.chan_lab.index('EMG')]
                # -----------------------------------------
                # Perform the exchange of eeg if neccessary
                # -----------------------------------------
                if exchange_eeg:
                    print "## Exchanging parietal and frontal EEG..."
                    tmp = eeg1
                    eeg1 = eeg2
                    eeg2 = eeg1
                # -----------------------------------
                # Extract features from these signals
                # -----------------------------------
                file_features = feature_extractor(eeg1,eeg2,emg,samples_per_epoch,ds_factor) 
                # --------------            
                # Extract labels
                # --------------
                num_intervals = np.shape(file_features)[0]
                raw_labels    = np.genfromtxt(data_folder+file_folder+'/'+str.rsplit(f,'.')[0]+'.STD', skip_header=0, dtype=str, comments="4s")    
                # ----------------      
                # Number of labels
                # ----------------
                nlabels = np.shape(raw_labels)[0]
                print "Number of labels is: " + str(nlabels)
                # --------------
                # Extract labels
                # --------------
                file_labels   = raw_labels[:,1:].copy()
                for i in range(nlabels):
                        file_labels[i,0] = mapping[raw_labels[i,1]]
                # ---------------------------------
                # Print out the number of artifacts
                # ---------------------------------
                tmp = file_labels[:,0]
                print "Number of artifacts is: " + str(np.count_nonzero(tmp[tmp=='-1']))
                # --------------------------------
                # Accumulating features and labels
                # --------------------------------
                features = np.vstack([features, file_features]) if np.shape(features)[0] else file_features
                labels   = np.vstack([labels, file_labels])     if np.shape(labels)[0]   else file_labels
                print "The shape of features is: " + str(np.shape(features))
                print "The shape of labels is: " + str(np.shape(labels))
                # ------------
                # Debug output
                # ------------
                print "File "+f+" processed and added to the dataset."
                print "--------------------------------------------------"
    # ---------------------
    # Feature normalization
    # ---------------------
    # use robust scaling
    scaler = preprocessing.RobustScaler()
    print "## The shape of final feature matrix is " + str(np.shape(features))
    # scaling slightly differs depending on the chosen feature extractor
    if feature_extractor_id=='temporal' or feature_extractor_id=='fourier':
        # get signal length
        signal_length = np.shape(features)[1]/3
        print "We have 3 channels, each of length: " + str(signal_length)
        # normalize corresponding channels
        features[:,0*signal_length:1*signal_length] = scaler.fit_transform(features[:,0*signal_length:1*signal_length])
        features[:,1*signal_length:2*signal_length] = scaler.fit_transform(features[:,1*signal_length:2*signal_length])
        features[:,2*signal_length:3*signal_length] = scaler.fit_transform(features[:,2*signal_length:3*signal_length])
        print "## The shape of final feature matrix is " + str(np.shape(features))
    # fourier spectrum energy bands features
    if feature_extractor_id=='fourier_energy':
        features = scaler.fit_transform(features)

    return [features,labels]




def temporal_raw_features(eeg1,eeg2,emg,samples_per_epoch,ds_factor):
    '''
    #Description: From given EEG/EMG data, extract and stack raw temporal epochs.
    ----------------------------------------------------------------------------------
    @param eeg1              => EEG1
    @param eeg2              => EEG2
    @param emg               => EMG
    @param samples_per_epoch => how many values we have per each signal interval/epoch
    @param ds_factor         => we downsample raw signals by this factor
     ---------------------------------------------------------------------------------
    @return => a matrix where each row contains raw signal of 'interval' length 
    ---------------------------------------------------------------------------------
    '''
    # ----------------------
    # Total number of epochs
    # ----------------------
    epochs = len(eeg1) / samples_per_epoch
    # -----------------------------------
    # Downsample the signals if specified
    # -----------------------------------
    eeg1 = signal.decimate(eeg1,ds_factor)
    eeg2 = signal.decimate(eeg2,ds_factor)
    emg  = signal.decimate(emg,ds_factor)
    samples_per_epoch = samples_per_epoch/ds_factor
    # -------------------------
    # Initialize feature matrix
    # -------------------------
    X = []
    # ------------------------
    # Construct feature matrix
    # ------------------------
    for i in range(int(epochs)):
        # get the signals of current epoch
        eeg1_epoch = eeg1[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        eeg2_epoch = eeg2[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        emg_epoch  = emg[int(samples_per_epoch)  * i: int(samples_per_epoch) * (i + 1)]
        # stack the signals into a feature vector
        features = np.stack( (eeg1_epoch,eeg2_epoch,emg_epoch), 0).flatten();
        # place feature vector into the feature matrix
        X = np.vstack([X, features]) if np.shape(X)[0] else features
    # return
    return X



def fourier_raw_features(eeg1,eeg2,emg,samples_per_epoch,ds_factor):
    '''
    #Description: 
    ----------------------------------------------------------------------------------
    @param eeg1              => EEG1
    @param eeg2              => EEG2
    @param emg               => EMG
    @param samples_per_epoch => how many values we have per each signal interval/epoch
    @param ds_factor         => we downsample fourier signals by this factor
     ---------------------------------------------------------------------------------
    @return => 
    ---------------------------------------------------------------------------------
    '''
    # ----------------------
    # Total number of epochs
    # ----------------------
    epochs = len(eeg1) / samples_per_epoch
    # -------------------------
    # Initialize feature matrix
    # -------------------------
    X = []
    # ------------------------
    # Construct feature matrix
    # ------------------------
    for i in range(int(epochs)):
        # Get signals of current epoch
        eeg1_epoch = eeg1[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        eeg2_epoch = eeg2[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        emg_epoch  = emg[int(samples_per_epoch)  * i: int(samples_per_epoch) * (i + 1)]
        # Convert them using FFT
        eeg1_pos_spectrum = np.abs(np.fft.rfft(eeg1_epoch)) ** 2
        eeg2_pos_spectrum = np.abs(np.fft.rfft(eeg2_epoch)) ** 2
        emg_pos_spectrum  = np.abs(np.fft.rfft(emg_epoch)) ** 2
        # downsample
        #print np.shape(eeg1_epoch)
        #eeg1_pos_spectrum = signal.decimate(eeg1_pos_spectrum,ds_factor)
        #eeg2_pos_spectrum = signal.decimate(eeg2_pos_spectrum,ds_factor)
        #emg_pos_spectrum  = signal.decimate(emg_pos_spectrum,ds_factor)
        # stack the signals into a feature vector
        features = np.stack( (eeg1_pos_spectrum[0:-1],eeg2_pos_spectrum[0:-1],emg_pos_spectrum[0:-1]), 0).flatten();
        # place feature vector into the feature matrix
        X = np.vstack([X, features]) if np.shape(X)[0] else features
    # return
    return X



def fourier_energy_features(eeg1,eeg2,emg,samples_per_epoch,ds_factor):

    # ----------------------
    # Total number of epochs
    # ----------------------
    epochs = len(eeg1) / samples_per_epoch
    # --------------------------------------------------------------------------- #
    # ---------------- Create array to represent frequency spectrum ------------- #
    # --------------------------------------------------------------------------- #
    # Compute the frequencies corresponding to the eeg_pos_spec indices. Because we
    # sample an entire interval of 'interval' waves at once, need to normalize
    # the frequencies by that factor as well.
    # Due to simetry we calculate only positive frequency spectrum, hence n/2+1
    eeg_pos_spectrum_len = int(samples_per_epoch / 2 + 1);
    eeg_f                = np.linspace(0, eeg_pos_spectrum_len - 1, eeg_pos_spectrum_len) / 4
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
        eeg1_pos_spectrum = np.abs(np.fft.rfft(eeg1_epoch)) ** 2
        eeg2_pos_spectrum = np.abs(np.fft.rfft(eeg2_epoch)) ** 2
        emg_pos_spectrum  = np.abs(np.fft.rfft(emg_epoch)) ** 2
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
    # efeatures = np.log(features);
    return features

'''
def mixed_features(file_path,interval_size=4,exchange_eeg=False, ds_factor=4):

    # get fourier features
    fftfeatures = fft_features(file_path,interval_size,exchange_eeg)
    # get raw temporal signal features
    rawfeatures = raw_signal_features(file_path,interval_size,exchange_eeg,ds_factor)
    # stack them into one matrix
    features = np.hstack([rawfeatures,fftfeatures])
    # return
    return features


def mel_features(file_path,interval_size=4,exchange_eeg=False):

    # Get Raw Data and Sampling Characteristics
    data              = load_edf(file_path)
    sample_rate       = data.sample_rate
    samples_per_epoch = interval_size * sample_rate
    eeg1 = data.X[data.chan_lab.index('EEG1')]
    eeg2 = data.X[data.chan_lab.index('EEG2')]
    emg  = data.X[data.chan_lab.index('EMG')]
    total_size = len(eeg1)
    epochs = total_size / samples_per_epoch
    # Create feature matrix from mel frequency spectrum
    X = []
    for i in range(int(epochs)):
        # Get signals of current epoch
        eeg1_epoch = eeg1[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        eeg2_epoch = eeg2[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
        emg_epoch  = emg[int(samples_per_epoch)  * i: int(samples_per_epoch) * (i + 1)]
        # Extract mel features
        matrix,features = fbank(eeg1_epoch,sample_rate,winlen=0.3,winstep=0.3)
        # Stack features
        X = np.vstack([X, features]) if np.shape(X)[0] else features

    scaler = preprocessing.RobustScaler()
    X = scaler.fit_transform(X)

    print X

    return X
'''


























