require 'nn'
require 'paths'
require 'cunn'
require 'gnuplot'
require 'lib.SGD'
require 'lib.MBGD'
local inout = require 'lib.inout'
local eval  = require 'lib.eval'

-------------------------------------------------------
-- Experiment variables
-------------------------------------------------------
retrain 	= false       -- Retrain or load the model
numExp 		= '5'        -- Number of experiment
augType		= '_aug'     -- Artifact augmentation type
maxIter		= 16	     -- Number of train iterations
extraIter 	= 0          -- Additional iterations
learnRate	= 0.01       -- Learning rate
learnRateDecay  = 0	     -- /(1+numIter*decay)
momentum	= 0.6        -- if stochastic gradient with momentum used
batchSize       = 50 	     -- size of the batch

--------------------------------------------------------
-- Constants: not to be changed in general
--------------------------------------------------------
numChan	 	= 1      -- 3 channels: EEG1;EEG2;EMG	
numLabels 	= 3	 -- number of testing data labels
epochSize 	= 13     -- length of epoch signal
cuda 		= true

--[[
--------------------------------------------------------
-- Architectural variables
--------------------------------------------------------
layer1 = 30
layer2 = 60
layer3 = 30

------------------------------------
-- Architecture of feature extractor
------------------------------------
ftextractor = nn.Sequential()
	      	:add(nn.Linear(epochSize,layer1))
		:add(nn.ReLU())
		:add(nn.Dropout(0.5))
		:add(nn.Linear(layer1,layer2))
		:add(nn.ReLU())
		:add(nn.View(-1))
------------------------------------
-- Full network
------------------------------------
net = nn.Sequential()
	:add(ftextractor)
	:add(nn.Dropout(0.5))
	:add(nn.Linear(layer2,layer3))
	:add(nn.ReLU())
	:add(nn.Dropout(0.5))
	:add(nn.Linear(layer3,2))
	:add(nn.LogSoftMax())
--]]
layer1 = 50
ftextractor = nn.Sequential()
	      	:add(nn.Linear(epochSize,layer1))
		:add(nn.ReLU())
		:add(nn.View(-1))
net = nn.Sequential()
	:add(ftextractor)
	:add(nn.Dropout(0.5))
	:add(nn.Linear(layer1,2))
	:add(nn.LogSoftMax())

------------------------------------------------------
-- DEBUG
------------------------------------------------------
print("## Network architecture:")
print(net:__tostring());
print("## The chosen experiment: "..numExp)
print("## The augmentation method: "..augType)
print("## Iterations: "..maxIter)
print("## Learning rate: "..learnRate)
print("## Learning rate decay: "..learnRateDecay)
print("## Layer 1: "..layer1)
--print("## Layer 2: "..layer2)
--print("## Layer 3: "..layer3)

-----------------------------------------------------
-- Load training data set
-----------------------------------------------------
if retrain or extraIter>0 then
	print("## Loading training data...")
	trainSet = inout.load_dataset('../../CSV/train_exp'..numExp..augType..'.csv',numChan,1)
	print("## Input dimensions: "..trainSet.data:size(1).." x "..trainSet.data:size(2))
	if cuda then
		trainSet.data  = trainSet.data:cuda()
		trainSet.label = trainSet.label:cuda()
	end
end

--------------------------------
-- In case we use existing model
--------------------------------
if not retrain then
	print('## Loading already saved model..')
        net = torch.load('models/fourier_binart')
end

---------------------------------------------------
-- Neural network optimization procedure
---------------------------------------------------
criterion = nn.ClassNLLCriterion()
trainer = nn.MBGD(net,criterion)
trainer.learningRate = learnRate
trainer.maxIteration = maxIter
trainer.learningRateDecay = learnRateDecay
trainer.momentum = momentum
trainer.batchSize = batchSize
if cuda then
	net = net:cuda()
	criterion = criterion:cuda()
end

---------------------------------------------------
-- Train the network or simply load it from the file
----------------------------------------------------
if retrain then
	print('## Training of the neural network begins...')
	collectgarbage()
    	timer = torch.Timer()
	trainer:train(trainSet)
	print('## Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')
else
	if extraIter>0 then
		print('## Performing additional '..extraIter..' iterations')
		trainer.maxIteration = extraIter
		trainer:train(trainSet)
	else
		print('## No additional iterations...')
	end
end

-----------------
-- Save the model
-----------------
torch.save('models/fourier_binart', net)
torch.save('models/fourier_extractor',ftextractor)

----------------------------------------------------------------
-- Load testing data sets from .CSV files of selected experiment
----------------------------------------------------------------
print("Loading validation data...:")
testSet = inout.load_dataset('../../CSV/test_exp'..numExp..'.csv',numChan,3)
print("Input dimensions: "..testSet.data:size(1)..'x'..testSet.data:size(2))
if cuda then
        testSet.data  = testSet.data:cuda()
        testSet.label = testSet.label:cuda()
end

----------------------------------------------------------------
-- Test the accuracy of the network on the testing data set
----------------------------------------------------------------
print "Validating artefakt detection..."
eval.test(testSet,net,false)
