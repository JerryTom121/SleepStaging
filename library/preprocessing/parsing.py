"""
Set of classes used for reading and parsing the data. Depending on the way 
the data is scored and recorded we implement several parsers.
"""
# Author: Djordje Miladinovic
# License:

import numpy as np
import pandas as pd

class SleepScoringParser(object):

	artifact_detection_mapping = []
	sleep_staging_mapping = []

	def parse(self,filepath):
		pass

	def get_binary_mapping(self):
		"""Returns:
			 A dictionary for - in case of artifact detection
		"""
		pass

	def get_4stage_mapping(self):
		"""Returns:
			A dictionary for mapping - in case of complete sleep staging
		"""
		pass

class SleepScoringParserUZH(SleepScoringParser):
	"""Class for parsing files given in UZH format
	"""

	def parse(self,filepath):
		"""Read labels given in UZH format.

		Args:
			filepath: path to .STD file containing labels

		Returns:
			Fetched labels given in Pandas DataFrame structure
		"""
		self.scoring = pd.DataFrame(np.genfromtxt(filepath, skip_header=0, dtype=str, comments="4s"))
		self.scoring.drop(self.scoring.columns[[0]], axis=1, inplace=True)  # drop unnecessary indices
		# keep only the first column in case there are more (e.g. from several human raters)
		self.scoring = pd.DataFrame(self.scoring[1])
		self.scoring.columns = ["label"]

		return self.scoring

	def get_binary_mapping(self):

		return { "w": +1, "n": +1, "r": +1,                               # normal
            	 "1": -1, "2": -1, "3": -1, "a": -1, "'": -1, "4": -1 }   # artifact

	def get_4stage_mapping(self):

		return { "w": 1, 										   # WAKE
			     "n": 2, 										   # NREM
			     "r": 3,                               			   # REM
		   		 "1": 4, "2": 4, "3": 4, "a": 4, "'": 4, "4": 4 }  # artifact

class SleepScoringParserUSZ(SleepScoringParser):
	"""Class for parsing files given in USZ.0 format"""

	def parse(self,filepath):
		"""Read labels given in USZ format.

		Args:
			filepath: path to .txt file containing labels

		Returns:
			Fetched labels given in Pandas DataFrame structure
		"""
		labels = []
		i = 0
		with open(filepath) as f:
			for line in f:
				i = i+1
				if i>18:
					labels.append(line.split()[0])
		self.scorings = pd.DataFrame(labels,columns=['label'])
		return self.scorings

	def get_binary_mapping(self):

		return { "AW": +1, "SWS": +1, "PS": +1,               # normal
            	 "AW-art": -1, "SWS-art": -1, "PS-art": -1}   # artifact

	def get_4stage_mapping(self):

		return { "AW":  1, 										# WAKE
			     "SWS": 2, 										# NREM
			     "PS":  3,                               		# REM
		   		 "AW-art": 4, "SWS-art": 4, "PS-art": 4}   	    # artifact


class SleepScoringParserAUT(SleepScoringParser):
	"""Class for parsing files given in the output 
	   format of our algorithm"""

	def parse(self,filepath):
		"""Read labels given in 1 column .csv file format.

		Args:
			filepath: path to .txt file containing labels

		Returns:
			Fetched labels given in Pandas DataFrame structure
		"""
		self.scorings = pd.read_csv(filepath,dtype=str)
		return self.scorings

	def get_binary_mapping(self):

		return { "1": +1, "2": +1, "3": +1,   # normal
            	 "4": -1}   				  # artifact

	def get_4stage_mapping(self):

		return { "1": 1, "2": 2, "3": 3, "4": 4} # idenitity mapping