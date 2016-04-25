'''
Library for dealing 
with Artefakts

		@Miladinovic Djordje 18.03.2016
'''
import numpy as np


def remove_artefakts(features,labels):
	'''
	Function used for removing all kind of
	Artefakts for "Artefakt free" generating
	of CSV files.
	'''
	print np.shape(features)
	print np.shape(labels)
	wake_art_ind = np.where(labels == 3)[0]
	nrem_art_ind = np.where(labels == 4)[0]
	rem_art_ind  = np.where(labels == 5)[0]
 	artefakt_indices = np.concatenate((wake_art_ind, nrem_art_ind,rem_art_ind), axis=0)
 	print "----------------------------------------"
 	print np.shape(wake_art_ind)
 	print np.shape(nrem_art_ind)
 	print np.shape(rem_art_ind)
 	print np.shape(artefakt_indices)
 	print "----------------------------------------" 
	# Delete Corresponding Features
	features = np.delete(features,artefakt_indices,0)
	# Delete Corresponding Label
	labels = np.delete(labels,artefakt_indices)
	print np.shape(features)
	print np.shape(labels)
	return [features,labels]



def stage_distribution(data_labels,sleep_stages):
	'''
	Function which returns frequency of occurence 
	for each stage in 'sleep_stages' from a given set 
	of labels in 'data_labels'.
	'''
	num_vals = data_labels.count()[0]

	frequencies = []
	for stage in sleep_stages:
		try:
			frequencies.append(data_labels.stack().value_counts(dropna=False)[stage]*1.0/num_vals)
		except:
			frequencies.append(0)
	
	return frequencies