""" 
Once the raw EEG/EMG data is processed and classified this class is
used to analyse obtained labels with respect to the overall 
distribution stages and the distribution of transitions among them.
"""
# Author: Djordje Miladinovic
# License: 

from library.preprocessing import SleepScoringParserUZH,SleepScoringParserUSZ,SleepScoringParserAUT
import pandas as pd
from sklearn import metrics


class SleepScoringAnalysis(object):
	"""Statistical analysis of sleep scoring results"""
	filepath = ""
	scoring = None
	parser = None

	def __init__(self, filepath, parsing_format):

		if parsing_format=="UZH":
			self.parser = SleepScoringParserUZH()

		elif parsing_format=="USZ":
			self.parser = SleepScoringParserUSZ()

		elif parsing_format=="AUT":
			self.parser = SleepScoringParserAUT()

		self.filepath = filepath
		self.scoring = self.parser.parse(filepath)

	# --------------------
	# Getters with mapping
	# --------------- ----
	def get_raw_label(self):
		""" Get raw scorings
		"""
		return self.scoring["label"]

	def get_4stage_label(self):
		""" Get 4 stage scorings
		"""
		mapped_scoring = self.scoring.replace({"label": self.parser.get_4stage_mapping()})
		return mapped_scoring["label"]

	def get_binary_label(self):
		""" Get binary scorings
		"""
		mapped_scoring = self.scoring.replace({"label": self.parser.get_binary_mapping()})
		return mapped_scoring["label"]

	# -----------------------
	# Label distribution info
	# -----------------------
	def percentage_raw_label(self):
		"""Calculate the presence percentage of each labeled stage.
		"""
		return self.scoring["label"].value_counts()/len(self.scoring)

	def percentage_4stage_label(self):
		"""Calculate the presence percentage of 4 main categories.
		"""
		mapped_scoring = self.scoring.replace({"label": self.parser.get_4stage_mapping()})
		return mapped_scoring["label"].value_counts()/len(mapped_scoring)

	def percentage_binary_label(self):
		"""Calculate the presence percentage of (non-)artifact data.
		"""
		mapped_scoring = self.scoring.replace({"label": self.parser.get_binary_mapping()})
		return mapped_scoring["label"].value_counts()/len(mapped_scoring)