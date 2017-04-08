"""A set of classes which implement different normalization
techniques. Different techniques correspond to different
feature extraction techniques."""

# Author: Djordje Miladinovic
# License:

import numpy as np

class _Normalizer(object):
    """Base function for feature normalization. The interface compiles
    with sklearn package."""

    def fit(self, features):
        """Implement in derived classes."""

    def transform(self, features):
        """Implement in derived classes."""

    def fit_transform(self, features):
        """Combines fit and transform into a single function.

        Parameters
        ----------
            features: a matrix of features to be transformed.

        Returns
        -------
            Transformed feature matrix."""

        self.fit(features)
        return self.transform(features)

class NormalizerZMUV(_Normalizer):
    """Normalize each channel independently with zero mean unit variance
    scaling method."""

    def __init__(self, num_channels, num_features):

        self.means = np.zeros(num_channels)
        self.stds = np.ones(num_channels)
        self.num_channels = num_channels
        self.num_features = num_features
        self.mean = 0
        self.std = 1

    def fit(self, features):
        """Calculate mean value and standard deviation of each component of
        the signal.

        Parameters
        ----------
            features: a matrix of features to be transformed."""

        for i in range(self.num_channels):
            self.means[i] = np.mean(features[:, i*self.num_features:(i+1)*self.num_features].flatten())
            self.stds[i] = np.std(features[:, i*self.num_features:(i+1)*self.num_features].flatten())
       
        print self.means
        print self.stds
        self.mean = np.mean(features)
        self.std = np.std(features)

    def transform(self, features):
        """Transform each signal component based on previously calculated
        parameters. We substruct the mean and divide by standard deviation.

        Parameters
        ----------
            features: a matrix of features to be transformed."""
        
        #for i in range(self.num_channels):
        #    features[:, i*self.num_features:(i+1)*self.num_features] = \
        #    (features[:, i*self.num_features:(i+1)*self.num_features]-self.means[i])/self.stds[i]

        features = (features-self.mean)/self.std

        print self.mean
        print self.std

        for i in range(self.num_channels):
            print np.shape(features[:, i*self.num_features:(i+1)*self.num_features])
            print np.mean(features[:, i*self.num_features:(i+1)*self.num_features].flatten())
            print np.std(features[:, i*self.num_features:(i+1)*self.num_features].flatten())

        return features
