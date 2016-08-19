import numpy as np

# Constants
ARTEFAKT_LABELS = ['1','2','3','a',"'",'4']
RATERS 		    = ['Christine','Andrea']

# To be changed if needed
#DATA_FOLDER     = '/home/djordje/Desktop/EEGData/DoubleScored/WildTypes/'
#FILES 		    = ['AS53B','AS53C','AS55B','AS55C','AS76D','AS76E','AS87H','AS87I']

DATA_FOLDER     = '/home/djordje/Desktop/EEGData/DoubleScored/Mutants/'
FILES 		    = ['AS52B','AS52C','AS54B','AS54C','AS73D','AS73E','AS75D','AS75E']

def stats(rater,labels):
	cnt = 0;
	for i in range(np.shape(labels)[0]):
		if labels[i][1] in ARTEFAKT_LABELS:
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


	andrea_labels = np.genfromtxt(DATA_FOLDER+'Andrea'+'/'+str.rsplit(f,'.')[0]+'.STD', skip_header=0, dtype=str, comments="4s") 
	a = stats('Andrea',andrea_labels)

	christine_labels = np.genfromtxt(DATA_FOLDER+'Christine'+'/'+str.rsplit(f,'.')[0]+'.STD', skip_header=0, dtype=str, comments="4s") 
	c = stats('Christine',christine_labels)


	nonart = 0;
	agr = 0;
	for i in range(np.shape(andrea_labels)[0]):
		if (andrea_labels[i][1] not in ARTEFAKT_LABELS) and (christine_labels[i][1] not in ARTEFAKT_LABELS):
			if andrea_labels[i][1]==christine_labels[i][1]:
				agr = agr+1;
			nonart = nonart+1;

	intersection_labels = andrea_labels.copy()
	for i in range(np.shape(intersection_labels)[0]):
		if (christine_labels[i][1] not in ARTEFAKT_LABELS) and (andrea_labels[i][1] in ARTEFAKT_LABELS):
			intersection_labels[i][1] = christine_labels[i][1]
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

