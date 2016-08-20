'''
Experiment class which is able to read experiment configuration 
from 'experiments.cfg' file from the same folder
'''
import ConfigParser
import ast



class Experiment(object):
    def __init__(self,exp_num):

    	# Read configuration file
        config = ConfigParser.ConfigParser()
        config.read('experiments.cfg')

        # Initialize some global variables
        self.eeg_folder = config.get('Global', 'EEG_Folder')
        self.csv_folder = config.get('Global', 'CSV_Folder')
        self.interval   = int(config.get('Global', 'Interval'))

        # Initialize experiment parameters
        self.split    = config.get('Experiment'+exp_num, 'split')=='True'
        self.rndsplit = config.get('Experiment'+exp_num, 'random_split')=='True'
        self.trainset = ast.literal_eval(config.get('Experiment'+exp_num, 'trainset'))
        self.testset  = ast.literal_eval(config.get('Experiment'+exp_num, 'testset'))
        self.extrain  = config.get('Experiment'+exp_num, 'exchange_EEG_trainset')=='True'
        self.extest   = config.get('Experiment'+exp_num, 'exchange_EEG_trainset')=='True'
        self.artrem   = config.get('Experiment'+exp_num, 'artifact_removal')=='True'
        self.feat_ext = config.get('Experiment'+exp_num, 'feature_extractor')