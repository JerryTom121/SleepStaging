'''
Script utilizes the files which are scored by more raters
and finds intersection in terms of artifacts.
Additionally, we print out statistics of comparison between 
labels of different raters.
'''

import numpy as np


# Constants
ARTIFACT_LABELS = ['1','2','3','a',"'",'4']
RATERS 		    = ['Christine','Andrea']
# -----------------------------------------


# Variables; To be changed when needed
# Wild types double scored data
#DATA_FOLDER     = '/home/djordje/Desktop/EEGData/DoubleScored/WildTypes/'
#FILES 		    = ['AS53B','AS53C','AS55B','AS55C','AS76D','AS76E','AS87H','AS87I']
# Mutants double scored data
DATA_FOLDER     = '/home/djordje/Desktop/EEGData/DoubleScored/Mutants/'
FILES 		    = ['AS52B','AS52C','AS54B','AS54C','AS73D','AS73E','AS75D','AS75E']
# ---------------------------------------------------------------------------------


def stats(rater,labels):
	'''
	Print out and return the number 
	of artifacts detected by the given rater
	with the given labels
	'''
	cnt = 0;
	for i in range(np.shape(labels)[0]):
		if labels[i][1] in ARTIFACT_LABELS:
			cnt = cnt + 1;
	print rater+': '+str(cnt) + '/' + str(np.shape(labels)[0])
	return cnt


a_sum = 0
c_sum = 0
inter_sum = 0
agr_sum = 0
nonart_sum = 0


for f in FILES:

	print "\nEvaluating file " + f

	# Get Andrea's labels
	andrea_labels = np.genfromtxt(DATA_FOLDER+'Andrea'+'/'+str.rsplit(f,'.')[0]+'.STD', skip_header=0, dtype=str, comments="4s") 
	a = stats('Andrea',andrea_labels)

	# Get Christine's labels
	christine_labels = np.genfromtxt(DATA_FOLDER+'Christine'+'/'+str.rsplit(f,'.')[0]+'.STD', skip_header=0, dtype=str, comments="4s") 
	c = stats('Christine',christine_labels)

	# Number of labels
	nlabels = np.shape(christine_labels)[0]

	# Calculate non-artifact agreement
	nonart = 0;
	agr    = 0;
	for i in range(np.shape(andrea_labels)[0]):
		if (andrea_labels[i][1] not in ARTIFACT_LABELS) and (christine_labels[i][1] not in ARTIFACT_LABELS):
			if andrea_labels[i][1]==christine_labels[i][1]:
				agr = agr+1;
			nonart = nonart+1;

	# Initial intersection matrix;
	# First column is enumeration
	# Second column are Andrea's labels and this is going to be changed
	# Third column are Andrea's labels
	# Fourth column are Christine's labels
	intersection_labels = np.hstack((
									 np.hstack((andrea_labels.copy(),
									            andrea_labels.copy()[:,1].reshape(nlabels,1))),
		                                        christine_labels.copy()[:,1].reshape(nlabels,1)))
	# One more column for the neighbourhood
	for i in range(nlabels):
		# fix intersection label if needed
		if (christine_labels[i][1] not in ARTIFACT_LABELS) and (andrea_labels[i][1] in ARTIFACT_LABELS):
			intersection_labels[i][1] = christine_labels[i][1]

	print intersection_labels
	inter = stats('Intersection',intersection_labels)
	np.savetxt(fname=DATA_FOLDER+'Intersection'+'/'+f+'.STD',X=intersection_labels,fmt='%s')

	print 'aretfakt agreement percentage: ' + str(inter/float(a+c-inter))
	print 'non-aretfakt agreement percentage: ' + str(agr/float(nonart))
	print 'artefakt corruption: ' + str(float(inter)/10800)

	a_sum = a_sum + a
	c_sum = c_sum + c
	inter_sum = inter_sum + inter
	agr_sum = agr_sum + agr
	nonart_sum = nonart_sum + nonart


print '--------------------------------------------------------------------'
print len(FILES)*10800
print "Andrea's artefakts: " + str(a_sum)
print "Christine's artefakts: " + str(c_sum)
print "Intersection artefakts: " + str(inter_sum)
print 'overall aretfakt agreement percentage: ' + str(inter_sum/float(a_sum+c_sum-inter_sum))
print 'overeall non-aretfakt agreement percentage: ' + str(agr_sum/float(nonart_sum))
print 'overall artefakt corruption: ' + str(float(inter_sum)/(len(FILES)*10800))

