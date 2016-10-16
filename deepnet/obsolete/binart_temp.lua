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
numExp 		= '1'        -- Number of experiment
augType		= '_rot_mir' -- Artifact augmentation type
maxIter		= 10	     -- Number of train iterations
extraIter 	= 0          -- Additional iterations
learnRate	= 0.002      -- Learning rate
learnRateDecay  = 0.01	     -- /(1+numIter*decay)
momentum	= 0.6        -- if stochastic gradient with momentum used
batchSize       = 10	     -- size of the batch

--------------------------------------------------------
-- Constants: not to be changed in general
--------------------------------------------------------
numChan	 	= 3      -- 3 channels: EEG1;EEG2;EMG	
numLabels 	= 3	 -- number of testing data labels
epochSize 	= 128    -- length of epoch signal
cuda 		= true

--------------------------------------------------------
-- Architectural variables
--------------------------------------------------------
-- Layer 1 (temporal convolution) parameters
CONV_L1_featureMaps = 6
CONV_L1_kernel      = 20
CONV_L1_stride      = 1
-- MaxPool 1
MP_L1_region 	    = 2
MP_L1_stride	    = 2
-- Layer 2 (temporal convolution) parameters
CONV_L2_featureMaps = 10
CONV_L2_kernel      = 10  --10
CONV_L2_stride      = 1
-- MaxPool 2
MP_L2_region        = 2
MP_L2_stride        = 2

-------------------------------------------------------
-- Convolutional neural network architecture
--------------------------------------------------------
convnet = nn.Sequential()
signal = epochSize
print(signal)
-- Add Layer 1:
convnet:add(nn.TemporalConvolution(numChan,CONV_L1_featureMaps,CONV_L1_kernel,CONV_L1_stride))
convnet:add(nn.ReLU()) 
signal  = (signal-CONV_L1_kernel)/CONV_L1_stride + 1 
-- Add Max pooling layer
--convnet:add(nn.TemporalMaxPooling(MP_L1_region,MP_L1_stride))
--signal =  (signal-MP_L1_region)/MP_L1_stride+1
--print(signal)
-- Add Layer 2
convnet:add(nn.TemporalConvolution(CONV_L1_featureMaps,CONV_L2_featureMaps,CONV_L2_kernel,CONV_L2_stride))
convnet:add(nn.ReLU())
signal  = (signal-CONV_L2_kernel)/CONV_L2_stride + 1 
-- Add Max pooling layer:
--convnet:add(nn.TemporalMaxPooling(MP_L2_region,MP_L2_stride))
--signal =  (signal-MP_L2_region)/MP_L2_stride+1
--print(signal)
-- Add Layer 3: fully connected layer
convnet:add(nn.View(CONV_L2_featureMaps*signal))

----------------------------
-- Full network architecture
---------------------------
net = nn.Sequential()
	:add(convnet)
	:add(nn.Dropout(0.5))
	:add(nn.Linear(CONV_L2_featureMaps*signal,50))
	:add(nn.ReLU())					-- :add(nn.Threshold(0, 1e-6)) instead of RELU?????
	:add(nn.Linear(50,2))
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
print("## Convolutional Layer 1 (feature maps,kernel): "..CONV_L1_featureMaps,CONV_L1_kernel)
print("## Convolutional Layer 2 (feature maps,kernel): "..CONV_L2_featureMaps,CONV_L2_kernel)

-----------------------------------------------------
-- Load training data set
-----------------------------------------------------
if retrain or extraIter>0 then
	print("## Loading training data...")
	trainSet = inout.load_dataset('../../CSV/train_exp'..numExp..augType..'.csv',numChan,1)
	print("## Input dimensions: "..trainSet.data:size(1)..'x'..trainSet.data:size(2))
	if cuda then
		trainSet.data  = trainSet.data:cuda()
		trainSet.label = trainSet.label:cuda()
	end
end

--------------------------------
---- In case we use existing model
----------------------------------
if not retrain then
        print('## Loading already saved model..')
        net = torch.load('models/temp_binart')
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
		-- Save the model
		torch.save('models/temp_binart', net)
	else
		print('## No additional iterations...')
	end

end

-----------------
-- Save the model
-----------------
torch.save('models/temp_binart', net)
torch.save('models/temp_extractor',convnet)

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
