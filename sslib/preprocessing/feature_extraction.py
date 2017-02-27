"""
Set of classes implementing different time series feature extraction
logics and from different file formats.
"""
# Author: Djordje Miladinovic
# License:

import numpy as np
from sslib.parsing import RecordingsParserUZH, RecordingsParserUSZ



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

    def __init__(self, recording_filepaths, interval_size=4):
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

    def _temporal_extractor(self, eeg1, eeg2, emg, sample_rate):
        """Get raw temporal signal from each epoch.
        """

        samples_per_epoch = int(self.interval_size*sample_rate)
        epochs = len(eeg1)/samples_per_epoch
        length = samples_per_epoch*epochs
        eeg1 = np.reshape(eeg1[0:length], (epochs, samples_per_epoch))
        eeg2 = np.reshape(eeg2[0:length], (epochs, samples_per_epoch))
        emg = np.reshape(emg[0:length], (epochs, samples_per_epoch))
        return np.hstack((eeg1, eeg2, emg))

    def _fourier_extractor(self):
        """Get raw fourier signal for each epoch.
        """
        #NYI

    def _energy_extractor(self, eeg1, eeg2, emg, sample_rate):
        """Get fourier spectral energy features for each epoch.
        """

        samples_per_epoch = int(self.interval_size*sample_rate)
        epochs = len(eeg1) / samples_per_epoch
        # Due to simetry we calculate only positive frequency spectrum, hence n/2+1
        eeg_pos_spectrum_len = int(samples_per_epoch / 2 + 1);
        eeg_f = np.linspace(0, eeg_pos_spectrum_len - 1, eeg_pos_spectrum_len) / 4
        # Create frequency buckets
        eeg1_bin_idx = []
        eeg2_bin_idx = []
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
        # EMG
        emg_bin = np.where(np.logical_and(4 <= eeg_f, eeg_f <= 40))[0]
        # - Calculate FT on 4s intervals and energy for each bucket within interval - #
        features = np.zeros((int(epochs), len(eeg1_bin_idx) + len(eeg2_bin_idx) + 1))
        for i in range(int(epochs)):
            # Get signals of current epoch
            eeg1_epoch = eeg1[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
            eeg2_epoch = eeg2[int(samples_per_epoch) * i: int(samples_per_epoch) * (i + 1)]
            emg_epoch  = emg[int(samples_per_epoch)  * i: int(samples_per_epoch) * (i + 1)]
            # Compute the FFT on the EEG  and EMG data
            eeg1_pos_spectrum = np.abs(np.fft.rfft(eeg1_epoch)) ** 2
            eeg2_pos_spectrum = np.abs(np.fft.rfft(eeg2_epoch)) ** 2
            emg_pos_spectrum  = np.abs(np.fft.rfft(emg_epoch)) ** 2
            # Compute the sums over the frequency buckets of the EEG signal.
            for j in range(len(eeg1_bin_idx)):
                features[i, j] = np.sum(eeg1_pos_spectrum[eeg1_bin_idx[j]])
            for j in range(len(eeg2_bin_idx)):
                features[i, len(eeg1_bin_idx)+j] = np.sum(eeg2_pos_spectrum[eeg2_bin_idx[j]])
            # The EMG power signal goes into a single bucket.
            features[i, -1] = np.sum(emg_pos_spectrum[emg_bin])
        return features

    def _get_features(self, fextractor):
        """Generic feature extraction from a set of files given in 'filepaths'.

        Parameters
        ----------
            fextractor: a method which implements concrete feature extractor.

        Returns
        -------
            A pile of stacked features.
        """

        [eeg1, eeg2, emg, srate] = self.parser.get_signals()
        return fextractor(eeg1, eeg2, emg, srate)

    def get_temporal_features(self):
        """Interface function for obtaining temporal features.
        """
        return self._get_features(self._temporal_extractor)

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

    def __init__(self, recording_filepaths, interval_size=4):
        super(FeatureExtractorUZH, self)\
                    .__init__(recording_filepaths, interval_size=4)
        self.parser = RecordingsParserUZH(self.recording_filepaths)

class FeatureExtractorUSZ(_FeatureExtractor):
    """A class for extracting features from files given in USZ format
    """

    def __init__(self, recording_filepaths, interval_size=4):
        super(FeatureExtractorUSZ, self)\
                    .__init__(recording_filepaths, interval_size=4)
        self.parser = RecordingsParserUSZ(self.recording_filepaths)
