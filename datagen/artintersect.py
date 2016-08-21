'''
Script utilizes the files which are scored by more raters and finds intersection in terms of artifacts.
Additionally, we print out statistics of comparison between labels of different raters.
'''

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------- CONSTANTS ------------------------------------- #
# --------------------------------------------------------------------------- #
ARTIFACTS = ['1','2','3','4','a',"'"]
# double scored wild types
WT_FOLDER = '/home/djordje/Desktop/EEGData/DoubleScored/WildTypes/'
WT_FILES  = ['AS53B','AS53C','AS55B','AS55C','AS76D','AS76E','AS87H','AS87I']
# double scored mutants
MT_FOLDER = '/home/djordje/Desktop/EEGData/DoubleScored/Mutants/'
MT_FILES  = ['AS52B','AS52C','AS54B','AS54C','AS73D','AS73E','AS75D','AS75E']
# human raters
RATERS    = ['Andrea','Christine']
# --------------------------------------------------------------------------- #
# --------------------------- Functions ------------------------------------- #
# --------------------------------------------------------------------------- #

def print_stats(rater,labels,arts):
	'''
	Print out and return the number 
	of artifacts detected by the given rater
	with the given labels
	'''
	cnt = 0;
	for i in range(np.shape(labels)[0]):
		if labels[i][1] in arts:
			cnt = cnt + 1;
	print rater+': '+str(cnt) + '/' + str(np.shape(labels)[0])
	return cnt


def find_intersection(root_folder,raters,files):
	'''
	Find intersection labels for given files
	@param root_folder the folder where all the raters folders are
	@param raters array specified rater names
	@param files name of the files which are double scored
	'''
	for file in files:

		# DEBUG
		print "----------------------"
		print "Evaluating file "+file
		print "----------------------"

		# Fetch and stack labels of given raters
		labels   = []
		for idx,rater in enumerate(raters):
			# get the labels of the current rater
			rater_labels = np.genfromtxt(root_folder+rater+'/'+file+'.STD', skip_header=0, dtype=str, comments="4s") 
			# DEBUG
			print_stats(rater,rater_labels,ARTIFACTS)
			# stack the labels to the final matrix
			labels = np.hstack([labels,rater_labels[:,1:2]])  if np.shape(labels)[0] else np.hstack([rater_labels,rater_labels[:,1:2]])

		# intersect labels
		for i in range(np.shape(labels)[0]):
			if (labels[i][2] in ARTIFACTS) and (labels[i][3] in ARTIFACTS): labels[i][1] = 'a' # artifact
			else: 															labels[i][1] = 'r' # non-artifact

		# DEBUG
		print_stats("Intersection",labels,['a'])
		print labels
		
		# save intersected labels
		np.savetxt(root_folder+'Intersection/'+file+'.STD',X=labels,fmt='%s')

# --------------------------------------------------------------------------- #
# ------------------- Perform label intersecting ---------------------------- #
# --------------------------------------------------------------------------- #

# Evaluate wild type files
print "Find intersection labels for wild types:"
find_intersection(WT_FOLDER,RATERS,WT_FILES)

# Evaluate mutants
print "Find intersection labels for mutants:"
find_intersection(MT_FOLDER,RATERS,MT_FILES)