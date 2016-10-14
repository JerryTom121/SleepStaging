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
retrain 	= false   -- Retrain or load the model
numExp 		= '9'    -- Number of experiment
augType		= '_rot' -- Artifact augmentation type
maxIter		= 10	 -- Number of train iterations
extraIter 	= 0      -- Additional iterations
learnRate	= 0.01   -- Learning rate
learnRateDecay  = 0	 -- /(1+numIter*decay)
momentum	= 0.6
batchSize       = 10

--------------------------------------------------------
-- Constants: not to be changed in general
--------------------------------------------------------
numChan	 	= 3      -- 3 channels: EEG1;EEG2;EMG	
numLabels 	= 3	 -- number of testing data labels
epochSize 	= 128    -- length of epoch signal
numFFTfeat      = 13     -- number of fourier features


--------------------------------------------------------
-- Architectural variables
--------------------------------------------------------
-- Convolution module
---------------------
-- Layer 1 (temporal convolution) parameters
CONV_L1_featureMaps = 20
CONV_L1_kernel      = 6 --20
CONV_L1_stride      = 2
-- MaxPool 1
MP_L1_region 	    = 4
MP_L1_stride	    = 2
-- Layer 2 (temporal convolution) parameters
CONV_L2_featureMaps = 15
CONV_L2_kernel      = 4  --10
CONV_L2_stride      = 1
-- MaxPool 2
MP_L2_region        = 3
MP_L2_stride        = 2

-- Fourier module
-----------------


-------------------------------------------------------
-- Neural network architecture
--------------------------------------------------------
-- Convolution module
---------------------
convnet = nn.Sequential()
signal = epochSize
print(signal)
-- Add Layer 1:
convnet:add(nn.TemporalConvolution(numChan,CONV_L1_featureMaps,CONV_L1_kernel,CONV_L1_stride))
convnet:add(nn.ReLU()) 
signal  = (signal-CONV_L1_kernel)/CONV_L1_stride + 1 
-- Add Max pooling layer
convnet:add(nn.TemporalMaxPooling(MP_L1_region,MP_L1_stride))
signal =  (signal-MP_L1_region)/MP_L1_stride+1
print(signal)
-- Add Layer 2
convnet:add(nn.TemporalConvolution(CONV_L1_featureMaps,CONV_L2_featureMaps,CONV_L2_kernel,CONV_L2_stride))
convnet:add(nn.ReLU())
signal  = (signal-CONV_L2_kernel)/CONV_L2_stride + 1 
-- Add Max pooling layer:
convnet:add(nn.TemporalMaxPooling(MP_L2_region,MP_L2_stride))
signal =  (signal-MP_L2_region)/MP_L2_stride+1
print(signal)
-- Add Layer 3: fully connected layer
convnet:add(nn.View(CONV_L2_featureMaps*signal))
-- Fourier module
-----------------
fftnet = nn.Sequential()
fftnet:add(nn.View(numFFTfeat))
-- Network construction
-----------------------
mix = nn.ParallelTable()
	 :add(convnet)
         :add(fftnet)
net = nn.Sequential()
	 :add(mix)
         :add(nn.JoinTable(1))
	 :add(nn.Dropout(0.5))
	 :add(nn.Linear(CONV_L2_featureMaps*signal+numFFTfeat,60))
	 :add(nn.ReLU())
	 :add(nn.Linear(60,2))
	 :add(nn.LogSoftMax())


------------------------------------------------------
-- DEBUG
------------------------------------------------------
print("-------------------------")
print("The network architecture:")
print("-------------------------")
print(net:__tostring());
print("-----------")
print("Parameters:")
print("-----------")
print("The chosen experiment: "..numExp)
print("The augmentation method: "..augType)
print("Iterations: "..maxIter)
print("Learning rate: "..learnRate)
print("Learning rate decay: "..learnRateDecay)
print("Convolutional Layer 1 (feature maps,kernel): "..CONV_L1_featureMaps,CONV_L1_kernel)
print("Convolutional Layer 2 (feature maps,kernel): "..CONV_L2_featureMaps,CONV_L2_kernel)


-----------------------------------------------------
-- Load training data set
-----------------------------------------------------
if retrain or extraIter>0 then

	print("------------------------")
	print("Loading training data...")
	print("------------------------")
	trainSet = inout.load_dataset('../../CSV/train_exp'..numExp..augType..'.csv',1,1)
	-- reshape data to fit network architecture
	print("------------------------------------------")
	print("Reformating data to fit different modules:")
	print("------------------------------------------")
	CONV_train = torch.reshape(trainSet.data[{{},{1,numChan*epochSize}}],
				   trainSet.data:size(1),numChan,epochSize):transpose(2,3)
	FFT_train  = trainSet.data[{{},{numChan*epochSize+1,numChan*epochSize+numFFTfeat}}]
	print("Conv module input dimensions: "..CONV_train:size(1).." x "..CONV_train:size(2).." x "..CONV_train:size(3))
	print("FFT module input dimensions:  "..FFT_train:size(1).." x "..FFT_train:size(2))
	-- Move to CUDA
	trainSet.convData = CONV_train:cuda()
	trainSet.fftData  = FFT_train:cuda() 
	trainSet.label    = trainSet.label:cuda()
end


---------------------------------------------------
-- Neural network optimization procedure
---------------------------------------------------
--criterion = nn.SoftMarginCriterion()
criterion = nn.ClassNLLCriterion()
trainer = nn.MBGD(net,criterion)
trainer.learningRate = learnRate
trainer.maxIteration = maxIter
trainer.learningRateDecay = learnRateDecay
trainer.momentum = momentum
trainer.batchSize = batchSize
net = net:cuda()
criterion = criterion:cuda()


---------------------------------------------------
-- Train the network or simply load it from the file
----------------------------------------------------
if retrain then
	print('----------------------------------------')
	print('Training of the neural network begins...')
	print('----------------------------------------')
	collectgarbage()
    	timer = torch.Timer()
	trainer:train(trainSet.convData,trainSet.fftData,trainSet.label)
	print('Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')
	-- Save the model
	torch.save('models/mixbinart'..numExp, net)
else
	print('---------------------------------')
        print('Loading already saved model')
	net = torch.load('models/mixbinart'..numExp)
	if extraIter>0 then
		print('----------------------')
		print('Performing additional '..extraIter..' iterations')
		trainer.maxIteration = extraIter
		trainer:train(trainSet.convData,trainSet.fftData,trainSet.label)
		-- Save the model
		torch.save('models/mixbinart'..numExp, net)
	else
		print('No additional iterations...')
	end

end


----------------------------------------------------------------
-- Load testing data sets from .CSV files of selected experiment
----------------------------------------------------------------
print("--------------------------")
print("Loading validation data...:")
print("--------------------------")
testSet  = inout.load_dataset('../../CSV/test_exp'..numExp..'.csv',1,numLabels)
-- reshape the validation data to fit our network
CONV_test = torch.reshape(testSet.data[{{},{1,numChan*epochSize}}],
			  testSet.data:size(1),numChan,epochSize):transpose(2,3)
FFT_test  = testSet.data[{{},{numChan*epochSize+1,numChan*epochSize+numFFTfeat}}]
-- move to CUDA
testSet.convData = CONV_test:cuda()
testSet.fftData  = FFT_test:cuda()
testSet.label = testSet.label:cuda()
print("Validation data loaded: contains "..testSet:size(1).." samples.")


----------------------------------------------------------------
-- Test the accuracy of the network on the testing data set
----------------------------------------------------------------
print "--------------------------------"
print "Validating artefakt detection..."
print "--------------------------------"
eval.test(testSet,net,true)
