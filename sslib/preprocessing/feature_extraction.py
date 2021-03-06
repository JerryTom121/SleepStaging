#
# Set of classes implementing different time series feature extraction
# logics and from different file formats.
#
# Author: Djordje Miladinovic
# License:
# -----------------------------------------------------------------------------

import numpy as np
from scipy import signal
from sslib.parsing import RecordingsParserUZH, RecordingsParserUSZ
import matplotlib.pyplot as plt
import math

class _FeatureExtractor(object):
    """Base class used for feature extraction. It is initialized with
    a list of paths which define a set of files. From the files we extract
    and stack features into a big feature matrix. Features are taken from by
    default 4 second slices. The feature matrix can be depicted as follows:
                 ____________________________________
                | EEG1 feat. | EEG2 feat. | EMG feat.|
                |            |            |          |
                |   ...      |    ...     |   ...    |
                |            |            |          |
                | EEG1 feat. | EEG2 feat. | EMG feat.|
                |____________|____________|__________|
    """

    def __init__(self, recording_filepaths, cpars, interval_size=4):
        """
        Parameters
        ----------
            recording_filepaths: path to each recording of the data set
            interval_size: eeg/emg signals are sliced into the periods of
            'interval_size' seconds and labeled accordingly.
        """

        self.recording_filepaths = recording_filepaths
        self.interval_size = interval_size
        self.parser = None
        self.cpars = cpars

    def _temporal_extractor(self, eeg1, eeg2, emg, sample_rate):
        """Get raw temporal signal from each epoch.
        """

        samples_per_epoch = int(self.interval_size*sample_rate)
        epochs = len(eeg1)/samples_per_epoch
        length = samples_per_epoch*epochs
        eeg1 = np.reshape(eeg1, (epochs, samples_per_epoch))
        eeg2 = np.reshape(eeg2, (epochs, samples_per_epoch))
        emg = np.reshape(emg, (epochs, samples_per_epoch))
        return np.hstack((eeg1, eeg2, emg))

    def _fourier_extractor(self, eeg1, eeg2, emg, sample_rate):
        """Get raw fourier signal for each epoch."""

        """
        samples_per_epoch = int(self.interval_size*sample_rate)
        epochs = len(eeg1)/samples_per_epoch
        F = np.zeros((epochs, (samples_per_epoch/2+1)*3))
        for i in range(int(epochs)):
            # Get signals of current epoch
            eeg1_epoch = eeg1[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
            eeg2_epoch = eeg2[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
            emg_epoch  = emg[int(samples_per_epoch)  * i: int(samples_per_epoch) * (i + 1)]
            # Compute the FFT on the EEG  and EMG data
            eeg1_pos_spectrum = np.abs(np.fft.rfft(eeg1_epoch)) ** 2
            eeg2_pos_spectrum = np.abs(np.fft.rfft(eeg2_epoch)) ** 2
            emg_pos_spectrum  = np.abs(np.fft.rfft(emg_epoch)) ** 2
            # Stack
            F[i,:] = np.hstack((eeg1_pos_spectrum, eeg2_pos_spectrum, emg_pos_spectrum))
        return F
        """

        window = 64 # must be even number...
        iters = len(eeg1)-window
        samples_per_epoch = int(self.interval_size*sample_rate)
        epochs = len(eeg1)/samples_per_epoch
        
        # Init
        eeg1_delta = np.zeros(iters)
        eeg1_theta = np.zeros(iters)
        eeg1_alpha = np.zeros(iters)
        eeg2_delta = np.zeros(iters)
        eeg2_theta = np.zeros(iters)
        eeg2_alpha = np.zeros(iters)
        emg_band = np.zeros(iters)
      
        # Identify important frequency bands
        nu = np.fft.rfftfreq(window, 1.0/sample_rate)
        delta_band = np.logical_and(nu>=0.5, nu<=4)
        theta_band = np.logical_and(nu>=6, nu<=9)
        alpha_band = np.logical_and(nu>9, nu<=15)
        all_bands = np.logical_and(nu>=4, nu<=40)


        for i in range(iters):
            # eeg1
            ft = np.absolute(np.fft.rfft(eeg1[i:i+window])) ** 2
            eeg1_delta[i] = np.sum(ft[delta_band])
       #     eeg1_theta[i] = math.sqrt(np.sum(ft[theta_band])/window*2)
       #     eeg2_alpha[i] = math.sqrt(np.sum(ft[alpha_band])/window*2)
            # eeg2
            ft = np.absolute(np.fft.rfft(eeg2[i:i+window])) ** 2
            eeg2_delta[i] = np.sum(ft[delta_band])
       #     eeg2_theta[i] = math.sqrt(np.sum(ft[theta_band])/window*2)
       #     eeg2_alpha[i] = math.sqrt(np.sum(ft[alpha_band])/window*2)
            # emg
            ft = np.absolute(np.fft.rfft(emg[i:i+window])) ** 2
            emg_band[i] = np.sum(ft[all_bands])

        # padding
        eeg1_delta = np.pad(eeg1_delta, pad_width=window/2, mode='edge')
        #eeg1_theta = np.pad(eeg1_theta, pad_width=window/2, mode='edge')
        #eeg1_alpha = np.pad(eeg1_alpha, pad_width=window/2, mode='edge')
        eeg2_delta = np.pad(eeg2_delta, pad_width=window/2, mode='edge')
        #eeg2_theta = np.pad(eeg2_theta, pad_width=window/2, mode='edge')
        #eeg2_alpha = np.pad(eeg2_alpha, pad_width=window/2, mode='edge')
        emg_band   = np.pad(emg_band, pad_width=window/2, mode='edge')
        
        # reshaping
        eeg1_delta = np.reshape(eeg1_delta, (epochs, samples_per_epoch))
        #eeg1_theta = np.reshape(eeg1_theta, (epochs, samples_per_epoch))
        #eeg1_alpha = np.reshape(eeg1_alpha, (epochs, samples_per_epoch))
        eeg2_delta = np.reshape(eeg2_delta, (epochs, samples_per_epoch))
        #eeg2_theta = np.reshape(eeg2_theta, (epochs, samples_per_epoch))
        #eeg2_alpha = np.reshape(eeg2_alpha, (epochs, samples_per_epoch))
        emg_band   = np.reshape(emg_band, (epochs, samples_per_epoch))
        
        # stack and return
        #return np.hstack((eeg1_delta, eeg1_theta, eeg1_alpha, eeg2_delta, eeg2_theta, eeg2_alpha, emg_band))

        #return np.hstack((eeg1_delta, eeg1_theta, eeg2_delta, eeg2_theta, emg_band))
        return np.hstack((eeg1_delta, eeg2_delta, emg_band))

    def _get_features(self, fextractor, normalize, filtering):
        """Generic feature extraction from a set of files given in 'filepaths'.

        Parameters
        ----------
            fextractor: a method which implements concrete feature extractor.

        Returns
        -------
            A pile of stacked features.
        """

        [eeg1, eeg2, emg, srate] = self.parser.get_signals(normalize, filtering)
        return fextractor(self.cpars[0]*eeg1, self.cpars[1]*eeg2, self.cpars[2]*emg, srate)

    def get_temporal_features(self, normalize, filtering):
        """Interface function for obtaining temporal features.
        """
        return self._get_features(self._temporal_extractor, normalize, filtering)

    def get_fourier_features(self):
        """Interface function for obtaining fourier features.
        """
        return self._get_features(self._fourier_extractor)

    def get_energy_features(self):
        """Interface function for obtaining energy features.
        """
        return self._get_features(self._energy_extractor)


class FeatureExtractorUZH(_FeatureExtractor):
    """A class for extracting features from files given in UZH format
    """

    def __init__(self, recording_filepaths, cpars, interval_size=4):
        super(FeatureExtractorUZH, self)\
                    .__init__(recording_filepaths, cpars, interval_size=4)
        self.parser = RecordingsParserUZH(self.recording_filepaths)
