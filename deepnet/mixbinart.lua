require 'nn'
require 'paths'
require 'csvigo'
require 'cunn'
require 'gnuplot'
require 'lib.SGD'
local inout = require 'lib.inout'
local eval  = require 'lib.eval'


-------------------------------------------------------
-- Experiment variables
-------------------------------------------------------
retrain 	= false   -- Retrain or load the model
numExp 		= '9'    -- Number of experiment
augType		= '_rot' -- Artifact augmentation type
maxIter		= 9	 -- Number of train iterations
extraIter 	= 0      -- Additional iterations
learnRate	= 0.0003-- Learning rate
learnRateDecay  = 0.2	 -- /(1+numIter*decay)

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
CONV_L1_featureMaps = 8
CONV_L1_kernel      = 20
CONV_L1_stride      = 1
-- Layer 2 (temporal convolution) parameters
CONV_L2_featureMaps = 12
CONV_L2_kernel      = 10
CONV_L2_stride      = 1
-- Output
CONV_output = 30
-- Fourier module
-----------------
-- Layer 1
FFT_L1 = 30
-- Output
FFT_output = 10


-------------------------------------------------------
-- Neural network architecture
--------------------------------------------------------
-- Convolution module
---------------------
convnet = nn.Sequential()
signal = epochSize
-- Add Layer 1:
convnet:add(nn.TemporalConvolution(numChan,CONV_L1_featureMaps,CONV_L1_kernel,CONV_L1_stride))
convnet:add(nn.ReLU()) 
signal  = (signal-CONV_L1_kernel)/CONV_L1_stride + 1 
-- Add Layer 2
convnet:add(nn.TemporalConvolution(CONV_L1_featureMaps,CONV_L2_featureMaps,CONV_L2_kernel,CONV_L2_stride))
convnet:add(nn.ReLU())
signal  = (signal-CONV_L2_kernel)/CONV_L2_stride + 1 
-- Add Layer 3: fully connected layer
convnet:add(nn.View(CONV_L2_featureMaps*signal))
convnet:add(nn.Linear(CONV_L2_featureMaps*signal,CONV_output))
convnet:add(nn.View(-1))
-- Fourier module
-----------------
fftnet = nn.Sequential()
-- Add Layer 1:
fftnet:add(nn.Linear(numFFTfeat,FFT_L1))
fftnet:add(nn.ReLU())
-- Output
fftnet:add(nn.Linear(FFT_L1,FFT_output))
fftnet:add(nn.View(-1))
-- Network construction
-----------------------
mix = nn.ParallelTable()
	 :add(convnet)
         :add(fftnet)
net = nn.Sequential()
	 :add(mix)
         :add(nn.JoinTable(1))
	 :add(nn.Linear(CONV_output+FFT_output,1))
         :add(nn.View(-1))


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
print("Convolutional Layer 1 (feature maps,kernel): "..CONV_L1_featureMaps,CONV_L1_kernel)
print("Convolutional Layer 2 (feature maps,kernel): "..CONV_L2_featureMaps,CONV_L2_kernel)
print("Convolutional network output: "..CONV_output)


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
criterion = nn.SoftMarginCriterion()
trainer = nn.SGD(net,criterion)
trainer.learningRate = learnRate
trainer.maxIteration = maxIter
trainer.learningRateDecay = learnRateDecay
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
	torch.save('models/binart'..numExp, net)
else
	print('---------------------------------')
        print('Loading already saved model')
	net = torch.load('models/binart'..numExp)
	if extraIter>0 then
		print('----------------------')
		print('Performing additional '..extraIter..' iterations')
		trainer = nn.SGD(net, criterion)
		trainer.learningRate = learnRate
		trainer.maxIteration = extraIter
		trainer.learningRateDecay = learnRateDecay
		trainer:train(trainSet.convData,trainSet.fftData,trainSet.label)
		-- Save the model
		torch.save('models/binart'..numExp, net)
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
