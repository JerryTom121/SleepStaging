"""
A script used for analysing and comparing results obtained from
human raters and the ones acquired from sleep scoring algorithm.
"""
# Author: Djordje Miladinovic
# License:

import config as cfg
from stats import SleepScoringAnalysis
from sklearn import metrics
import numpy as np


def output_label_distribution(filepath, parser, comment):
    """Auxiliary function used to output the percentage
    of each label of a given file.

    Args:
            filepath: path to file containing labels
            parser: the parser specific to the format of
                    the file which contain labels.
                    options are: UZH,USZ,AUT
            expname: a comment on what is being tested
    """
    print " ----------------------------------------- "
    print " -- Stage distribution: " + comment
    print " ----------------------------------------- "
    ssa = SleepScoringAnalysis(filepath, parser)
    print ssa.percentage_raw_label()
    print ssa.percentage_4stage_label()
    print ssa.percentage_binary_label()


def output_predeval(truth_filepath, truth_parser, preds_filepath,\
                    preds_parser, comment):
    """Auxiliary function used to evaluate and output predictions
    compared to the ground truth w.r.t several metrics.

    Args:
            truth_filepath: path to file containing true labels
            truth_parser:  the parser specific to the format of
                    the file which contain labels.
                    options are: UZH,USZ,AUT
            preds_filepath: path to file containing predictions
            preds_parser:  the parser specific to the format of
                    the file which contain labels.
                    options are: UZH,USZ,AUT
            comment: a comment on what is being tested
    """
    ssa_truth = SleepScoringAnalysis(truth_filepath, truth_parser)
    ssa_preds = SleepScoringAnalysis(preds_filepath, preds_parser)
    length = min(ssa_truth.get_raw_label().size, ssa_preds.get_raw_label().size)
    print "\n"
    print "----------------------------------------"
    print "-- " + comment
    cmat_ss = metrics.confusion_matrix(ssa_truth.get_4stage_label()[0:length],\
                                       ssa_preds.get_4stage_label()[0:length])
    cmat_artdet = \
          metrics.confusion_matrix(ssa_truth.get_binary_label()[0:length], \
                                   ssa_preds.get_binary_label()[0:length])
    print "----------------------------------------"
    print "| INFO: Artifact percentage = " + \
           format((cmat_artdet[0, 1]+cmat_artdet[0, 0])*100.0/ \
                   np.sum(np.sum(cmat_artdet, axis=0), axis=0), '.2f') \
           + "%"
    print "----------------------------------------"
    print "| EVAL: Sleep scoring confusion matrix:"
    print cmat_ss
    print "----------------------------------------"
    print "| EVAL: Artifact detection confusion matrix:"
    print cmat_artdet
    print "----------------------------------------"
    print "| EVAL: Artifact detection evaluation:"
    print "| Accuracy: " + \
    format(metrics.accuracy_score(ssa_truth.get_binary_label()[0:length], \
                               ssa_preds.get_binary_label()[0:length]), '.2f')
    print "| Recall: " + \
    format(metrics.recall_score(ssa_truth.get_binary_label()[0:length], \
                             ssa_preds.get_binary_label()[0:length]), '.2f')
    print "| Precision: "+\
    format(metrics.precision_score(ssa_truth.get_binary_label()[0:length], \
                                ssa_preds.get_binary_label()[0:length]), '.2f')
    print "----------------------------------------"
    print "| EVAL: No artifact sleep scoring accuracy:"
    print "| "+format((cmat_ss[0, 0]+cmat_ss[1, 1]+cmat_ss[2, 2])*1.0/ \
                      (cmat_ss[0, 0]+cmat_ss[1, 1]+cmat_ss[2, 2]+ \
                       cmat_ss[1, 0]+cmat_ss[2, 0]+cmat_ss[0, 1]+ \
                       cmat_ss[0, 2]+cmat_ss[1, 2]+cmat_ss[2, 1]), '.2f')
    print "\n"

# --------------------------------------------------------------------------- #
# -------------------------- Ground truth vs predictions -------------------- #
# --------------------------------------------------------------------------- #
np.set_printoptions(precision=2, suppress=True)

# Test trial 1 from USZ
output_predeval(cfg.EEG_FOLDER+"AnotherLab/labels/Trial1_events.txt", "USZ",\
	            cfg.PREDS_FOLDER+"Trial1_1.csv", "AUT", \
	            "Testing trial 1_1 from USZ")

# Test trial 2 from USZ
output_predeval(cfg.EEG_FOLDER+"AnotherLab/labels/Trial2_events.txt", "USZ", \
	            cfg.PREDS_FOLDER+"Trial2_1.csv", "AUT", \
	            "Testing trial 2_1 from USZ")

# Test a random file from UZH
output_predeval(cfg.EEG_FOLDER+"EEGDataPool/CM/GABRA/WT/GAB2R22B.STD", "UZH", \
	            cfg.PREDS_FOLDER+"GAB2R22B.csv", "AUT", \
	            "Testing GAB2R22B from UZH")

# Test artifact double scored data from UZH - wild types
output_predeval(cfg.EEG_FOLDER+\
				"DoubleScored/WildTypes/Intersection/Test/AS76E.STD", "UZH", \
	            cfg.PREDS_FOLDER+"AS76E.csv", "AUT", \
	            "Testing DS WT AS76E from UZH")

output_predeval(cfg.EEG_FOLDER+\
	            "DoubleScored/WildTypes/Intersection/Test/AS87H.STD", "UZH", \
	            cfg.PREDS_FOLDER+"AS87H.csv", "AUT", \
	            "Testing DS WT AS87H from UZH")

# Test artifact double scored data from UZH - mutants
output_predeval(cfg.EEG_FOLDER+\
                "DoubleScored/Mutants/Intersection/Test/AS73E.STD", "UZH", \
                cfg.PREDS_FOLDER+"AS73E.csv", "AUT", \
                "Testing DS WT AS73E from UZH")

output_predeval(cfg.EEG_FOLDER+\
                "DoubleScored/Mutants/Intersection/Test/AS75D.STD", "UZH", \
                cfg.PREDS_FOLDER+"AS75D.csv", "AUT", \
                "Testing DS WT AS75D from UZH")
