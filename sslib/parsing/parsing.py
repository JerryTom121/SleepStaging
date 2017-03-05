"""
Set of classes used for reading and parsing the data. Depending on the way
the data is scored and recorded we implement several parsers.
"""
# Author: Djordje Miladinovic
# License:

import numpy as np
import pandas as pd
from sslib.external.edfplus import load_edf

# --------------------------------------------------------------------------- #
# --- Parsing of the files which contain raw data --------------------------- #
# --------------------------------------------------------------------------- #
class _RecordingsParser(object):
    """Base class for parsing files with EEG and EMG recordings.

    Warning: This class should not be used directly.
        Use derived classes instead.
    """
    def __init__(self, filepaths):
        """
        Parameters
        ----------
        filepaths: a list or a single file path to the recording to be parsed.
        """

        if isinstance(filepaths, basestring):
            self.filepaths = [filepaths]
        else:
            self.filepaths = filepaths
            self.filepaths.sort()

    def get_signals(self):
        """To be implemented in derived classes
        """

class RecordingsParserUZH(_RecordingsParser):
    """Class for parsing files given in UZH format.
    """
    def get_signals(self):
        """
        """

        eeg1 = np.ndarray(0)
        eeg2 = np.ndarray(0)
        emg = np.ndarray(0)

        for filepath in self.filepaths:

            data = load_edf(filepath)
            srate = data.sample_rate

            eeg1 = np.hstack([eeg1, data.X[data.chan_lab.index('EEG1')]]) \
            if np.shape(eeg1)[0] else data.X[data.chan_lab.index('EEG1')]
            eeg2 = np.hstack([eeg2, data.X[data.chan_lab.index('EEG2')]]) \
            if np.shape(eeg2)[0] else data.X[data.chan_lab.index('EEG2')]
            emg = np.hstack([emg, data.X[data.chan_lab.index('EMG')]]) \
            if np.shape(emg)[0] else data.X[data.chan_lab.index('EMG')]

            print "## Recording " + filepath + " parsed"

        return [eeg1, eeg2, emg, srate]

class RecordingsParserUSZ(_RecordingsParser):
    """Class for parsing files given in USZ format.
    """
    def get_signals(self):
        """
        """

        eeg1 = np.ndarray(0)
        eeg2 = np.ndarray(0)
        emg = np.ndarray(0)

        for filepath in self.filepaths:

            data = load_edf(filepath)
            srate = int(round(data.sample_rate))
            eeg1 = np.hstack([eeg1, data.X[data.chan_lab.index('Right-1')]]) \
            if np.shape(eeg1)[0] else data.X[data.chan_lab.index('Right-1')]
            eeg2 = np.hstack([eeg2, data.X[data.chan_lab.index('Left-1')]]) \
            if np.shape(eeg2)[0] else data.X[data.chan_lab.index('Left-1')]
            emg = np.hstack([emg, data.X[data.chan_lab.index('EMG-1')]]) \
            if np.shape(emg)[0] else data.X[data.chan_lab.index('EMG-1')]

            print "## Recording " + filepath + " parsed"

        return [eeg1, eeg2, emg, srate]



# --------------------------------------------------------------------------- #
# --- Parsing of the files which contain sleep scoring results -------------- #
# ---   - manually or automatically labeled --------------------------------- #
# --------------------------------------------------------------------------- #
class _ScoringParser(object):
    """Base class for parsing scorings of EEG and EMG recordings.

    Warning: This class should not be used directly.
        Use derived classes instead.
    """

    def __init__(self, filepaths):
        """
        Parameters
        ----------
                filepaths: a list of paths to scorings for parsing.
        """

        if isinstance(filepaths, basestring):
            self.filepaths = [filepaths]
        else:
            self.filepaths = filepaths
            self.filepaths.sort()


    def _parse(self, filepath):
        """abstract
        """
    def _get_binary_mapping(self):
        """abstract
        """
    def _get_4stage_mapping(self):
        """abstract
        """

    def get_raw_scorings(self):
        """To be implemented in derived classes
        """

        scorings = pd.DataFrame(columns=['label'])

        for filepath in self.filepaths:
            scoring = self._parse(filepath)
            scorings = pd.concat([scorings, scoring])
            print "## Scoring " + filepath + " parsed"

        return scorings


    def get_binary_scorings(self):
        """Get the scorings for artifact detection.
        """

        scorings = self.get_raw_scorings()
        mapped_scorings = \
        scorings.replace({"label": self._get_binary_mapping()})
        return np.array(mapped_scorings)

    def get_4stage_scorings(self):
        """Get the scorings for full sleep staging.
        """

        scorings = self.get_raw_scorings()
        mapped_scorings = \
        scorings.replace({"label": self._get_4stage_mapping()})
        return np.array(mapped_scorings)


class ScoringParserUZH(_ScoringParser):
    """Class for parsing files given in UZH format
    """

    def _parse(self, filepath):
        """Read labels given in UZH format.

        Parameters
        ----------
            filepath: path to .STD file containing labels

        Returns
        -------
            Fetched labels given in Pandas DataFrame structure
        """

        scorings = pd.DataFrame(\
            np.genfromtxt(filepath, skip_header=0, dtype=str, comments="4s"))
        scorings.drop(scorings.columns[[0]], axis=1, inplace=True)

        # keep only the first column in case of double scoring
        scorings = pd.DataFrame(scorings[1])
        scorings.columns = ["label"]
        return scorings

    def _get_binary_mapping(self):

        return {"w": 2, "n": 2, "r": 2,                         # regular
                "1": 1, "2": 1, "3": 1, "a": 1, "'": 1, "4": 1, # artifact
                "U": 5}                                         # uknown/ambigious

    def _get_4stage_mapping(self):

        return {"w": 1,                                         # WAKE
                "n": 2,                                         # NREM
                "r": 3,                                         # REM
                "1": 4, "2": 4, "3": 4, "a": 4, "'": 4, "4": 4, # artifact
                "U": 5}                                         # uknown/ambigious

class ScoringParserUSZ(_ScoringParser):
    """Class for parsing files given in USZ.0 format"""

    def _parse(self, filepath):
        """Read labels given in USZ format.

        Parameters
        ----------
            filepath: path to .txt file containing labels

        Returns
        -------
            Fetched labels given in Pandas DataFrame structure
        """

        labels = []
        i = 0
        with open(filepath) as file_:
            for line in file_:
                i = i+1
                if i > 18:
                    labels.append(line.split()[0])
        scorings = pd.DataFrame(labels, columns=['label'])
        return scorings

    def _get_binary_mapping(self):

        return {"AW": 2, "SWS": 2, "PS": 2,             # regular
                "AW-art": 1, "SWS-art": 1, "PS-art": 1} # artifact

    def _get_4stage_mapping(self):

        return {"AW":  1,                               # WAKE
                "SWS": 2,                               # NREM
                "PS":  3,                               # REM
                "AW-art": 4, "SWS-art": 4, "PS-art": 4} # artifact
