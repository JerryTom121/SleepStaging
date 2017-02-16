"""A set of classes which implement different normalization
techniques. Different techniques correspond to different
feature extraction techniques.
"""
# Author: Djordje Miladinovic
# License:

import numpy as np

class _Normalizer(object):
    """Base function for feature normalization. The interface compiles
    with sklearn package.
    """

    def fit(self, features):
        """Implement in derived classes.
        """

    def transform(self, features):
        """Implement in derived classes.
        """

    def fit_transform(self, features):
        """Combines fit and transform into a single function.

        Parameters
        ----------
            features: a matrix of features to be transformed.

        Returns
        -------
            Transformed feature matrix.
        """
        self.fit(features)
        return self.transform(features)

class NormalizerTemporal(_Normalizer):
    """Normalize eeg1, eeg2, emg signals independently with sklearn robust
    scaling method. Each signal is occuppies one third of feature matrix.
    """

    def __init__(self):
        # eeg1 pars
        self.eeg1_mean = 0
        self.eeg1_std = 1
        # eeg2 pars
        self.eeg2_mean = 0
        self.eeg2_std = 1
        # eeg3 pars
        self.emg_mean = 0
        self.emg_std = 1


    def fit(self, features):
        """Calculate mean value and standard deviation of each component of
        the signal. Updates the state of the object.

        Parameters
        ----------
            features: a matrix of features to be transformed.
        """
        ncols = np.shape(features)[1]/3
        # Get parameters for eeg1
        self.eeg1_mean = np.mean(features[:, 0:ncols].flatten())
        self.eeg1_std = np.std(features[:, 0:ncols].flatten())
        # Get parameters for eeg2
        self.eeg2_mean = np.mean(features[:, ncols:2*ncols].flatten())
        self.eeg2_std = np.std(features[:, ncols:2*ncols].flatten())
        # Get parameters for emg
        self.emg_mean = np.mean(features[:, 2*ncols:3*ncols].flatten())
        self.emg_std = np.std(features[:, 2*ncols:3*ncols].flatten())

    def transform(self, features):
        """Transform each signal component based on previously calculated
        parameters. We substruct the mean and divide by standard deviation.

        Parameters
        ----------
            features: a matrix of features to be transformed.
        """

        ncols = np.shape(features)[1]/3

        features[:, 0:ncols] \
        = (features[:, 0:ncols] - self.eeg1_mean)/self.eeg1_std

        features[:, ncols:2*ncols] \
        = (features[:, ncols:2*ncols] - self.eeg2_mean)/self.eeg2_std

        features[:, 2*ncols:3*ncols] \
        = (features[:, 2*ncols:3*ncols] - self.emg_mean)/self.emg_std

        return features
