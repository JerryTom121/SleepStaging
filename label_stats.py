import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
np.set_printoptions(suppress=True)

######### Parameters, adapt if necessary #############
labelPath = './data/scorings'	# can split into test and train
SCORER = 3	# 1: double scores, 2: scorer 1, : scorer 2
IGNORE_ARTIFACTS_AND_UNDECIDED = True # ignore artifacts and undecided scores for length of episodes
##################################


# count labels
# w, n, r, a, U
counts = [0,0,0,0,0]

# episode - lengths
episodes = [[],[],[],[],[]]


# phase changes
changes = np.zeros(shape=(5,5))

# in case of undecided score: pairs of score
undecidedPairs = dict()

def labelToInt(x):
	if x == 'w': return 0
	elif x == 'n': return 1
	elif x == 'r': return 2
	elif x == 'a': return 3
	elif x == 'U': return 4
	else: return 3 # all the number labels, all unknown labels


# for each file, get the necessary counts
def processFile(path):
	with open(path,'r') as f:
		state = labelToInt(f.readline().split()[SCORER])
		counts[state] += 1
		length = 1

		for line in f:
			label = line.split()[SCORER]
			# simple count
			counts[labelToInt(label)] += 1
			# handle undecided double scores
			if label == 'U':
				# create key
				both = '-'.join(sorted(line.split()[2:4]))
				# increase value by one, create key if necessary
				c = undecidedPairs.setdefault(both, undecidedPairs.get(both,0))
				undecidedPairs[both] = c+1
			# state transition
			if labelToInt(label) != state:
				if IGNORE_ARTIFACTS_AND_UNDECIDED:
					if labelToInt(label) <= 2: # no state transition for artifacts and undecided labels
						changes[state, labelToInt(label)] += 1
						state = labelToInt(label)
						episodes[labelToInt(label)].append(length)
						# print(length)
						# print("")
						length = 1
					else:
						length += 1
				
				else: # Artifacts and Undecided labels count as episodes
					# Sanity check
					# print("state: " + str(state))
					# print("label: " + str(labelToInt(label)))

					changes[state, labelToInt(label)] += 1
					state = labelToInt(label)
					episodes[labelToInt(label)].append(length)
					# print(length)
					# print("")
					length = 1
			else:
				length += 1

# Process files line by line
for root, dirs, files in os.walk(labelPath):
	for name in files:
		path = os.path.join(root, name)
		print(path)
		processFile(path)


# calculate stats
print("")
print("Total number of labels")
print(np.sum(np.array(counts)))

print("Counts for each label")
print("w,n,r,a,U")
print(counts)
print("Counts for pairs of undecided labels")
print(undecidedPairs)
print("Transition matrix")
print(changes)

wEpisodes = np.array(episodes[0])
nEpisodes = np.array(episodes[1])
rEpisodes = np.array(episodes[2])
aEpisodes = np.array(episodes[3])
uEpisodes = np.array(episodes[4])

wMean = np.mean(wEpisodes)
print("Mean w episode: {}".format(wMean))
nMean = np.mean(nEpisodes)
print("Mean n episode: {}".format(nMean))
rMean = np.mean(rEpisodes)
print("Mean r episode: {}".format(rMean))

if IGNORE_ARTIFACTS_AND_UNDECIDED:
	aMean = 0 # avoid error for mean of empty array
else: 
	aMean = np.mean(aEpisodes)
print("Mean a episode: {}".format(aMean))

if SCORER != 1 or IGNORE_ARTIFACTS_AND_UNDECIDED:
	uMean = 0; # avoid error for mean of empty array
else: 
	uMean = np.mean(uEpisodes)
print("Mean u episode: {}".format(uMean))

meanEpisodelength = \
	(	wMean * wEpisodes.shape[0] +
		nMean * nEpisodes.shape[0] +
		rMean * rEpisodes.shape[0] + 
		aMean * aEpisodes.shape[0] +
		uMean * uEpisodes.shape[0] ) / \
	(	wEpisodes.shape[0] +
		nEpisodes.shape[0] +
		rEpisodes.shape[0] + 
		aEpisodes.shape[0] +
		uEpisodes.shape[0] )

print("mean episode: {}".format(meanEpisodelength))

#sanity check
# print("total nunber of episodes: {}".format(np.sum(changes)))
# print("Alleged number of labels: {}".format(np.sum(changes) * meanEpisodelength))

#TODO: print figures