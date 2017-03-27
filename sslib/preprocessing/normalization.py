"""A set of classes which implement different normalization
techniques. Different techniques correspond to different
feature extraction techniques.
"""
# Author: Djordje Miladinovic
# License:

import numpy as np
import sklearn

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

class NormalizerZMUV(_Normalizer):
    """Normalize eeg1, eeg2, emg signals independently with sklearn robust
    scaling method. Each signal is occuppies one third of feature matrix.
    """

    def __init__(self, num_channels):
        self.means = np.zeros(num_channels)
        self.stds = np.ones(num_channels)
        self.num_channels = num_channels


    def fit(self, features):
        """Calculate mean value and standard deviation of each component of
        the signal. Updates the state of the object.

        Parameters
        ----------
            features: a matrix of features to be transformed.
        """
        for i in range(self.num_channels):
            self.means[i] = np.mean(features[:, i*self.num_channels:(i+1)*self.num_channels].flatten())
            self.stds[i] = np.std(features[:, i*self.num_channels:(i+1)*self.num_channels].flatten())

    def transform(self, features):
        """Transform each signal component based on previously calculated
        parameters. We substruct the mean and divide by standard deviation.

        Parameters
        ----------
            features: a matrix of features to be transformed.
        """
        for i in range(self.num_channels):
            features[:, i*self.num_channels:(i+1)*self.num_channels] = (features[:, i*self.num_channels:(i+1)*self.num_channels]-self.means[i])/self.stds[i]

        return features

class NormalizerEnergy(_Normalizer):

    def __init__(self):
        self.scaler = sklearn.preprocessing.StandardScaler()

    def fit(self,features):
        self.scaler.fit(features)

    def transform(self,features):
        return self.scaler.transform(features)
