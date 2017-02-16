"""
Set of classes implementing different time series feature extraction
logics and from different file formats.
"""
# Author: Djordje Miladinovic
# License:

import numpy as np
from sslib.parsing import RecordingsParserUZH, RecordingsParserUSZ



class _FeatureExtractor(object):
    """Based class used for feature extraction. It is initialized with
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
        eeg1 = np.reshape(eeg1, (epochs, samples_per_epoch))
        eeg2 = np.reshape(eeg2, (epochs, samples_per_epoch))
        emg = np.reshape(emg, (epochs, samples_per_epoch))
        return np.hstack((eeg1, eeg2, emg))

    def _fourier_extractor(self):
        """Get raw fourier signal for each epoch.
        """
        # NYI

    def _energy_extractor(self):
        """Get fourier spectral energy features for each epoch.
        """
        # NYI

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
