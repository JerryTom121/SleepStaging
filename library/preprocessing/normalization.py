'''
Classes used for feature normalization

@djordjem 10:11:2016
'''
import numpy as np
from sklearn import preprocessing

class Normalizer:

	def fit(self, X):
		pass

	def transform(self, X):
		return X

	def fit_transform(self, X):

		# fit
		self.fit(X)
		# transform
		return self.transform(X)


class RawDataNormalizer(Normalizer):
	'''
	Normalize 3 channels (2 EEG and 1 EMG) independently
	'''
	def __init__(self):
		self.eeg1_scaler = preprocessing.RobustScaler()
		self.eeg2_scaler = preprocessing.RobustScaler()
		self.emg_scaler  = preprocessing.RobustScaler()

	def fit(self,X):
		# length of EEG/EMG signals
		slength = np.shape(X)[1]/3
		# normalize corresponding channels
		self.eeg1_scaler.fit(X[:,0*slength:1*slength])
		self.eeg2_scaler.fit(X[:,1*slength:2*slength])
		self.emg_scaler.fit(X[:,2*slength:3*slength])

	def transform(self,X):
		# length of EEG/EMG signals
		slength = np.shape(X)[1]/3
		# normalize corresponding channels
		X[:,0*slength:1*slength] = self.eeg1_scaler.transform(X[:,0*slength:1*slength])
		X[:,1*slength:2*slength] = self.eeg2_scaler.transform(X[:,1*slength:2*slength])
		X[:,2*slength:3*slength] = self.emg_scaler.transform(X[:,2*slength:3*slength])

		return X

class FourierEnergyNormalizer(Normalizer):

	def __init__(self):
		self.scaler = preprocessing.RobustScaler()

	def fit(self,X):
		self.scaler.fit(X)

	def transform(self,X):
		return self.scaler.transform(X)
