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
retrain 	= true   -- Retrain or load the model
numExp 		= '9'    -- Number of experiment
augType		= '_rot' -- Artifact augmentation type
maxIter		= 5	 -- Number of train iterations
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


---------------------------------------------------------
-- Construct the network
---------------------------------------------------------
fourier_extractor = torch.load('models/fourier_extractor')
tempcon_extractor = torch.load('models/temp_extractor')

net = nn.Sequential()
	:add(nn.ParallelTable()
	       :add(fourier_extractor)
	       :add(tempcon_extractor)	
	)
	:add(nn.JoinTable(1))
	:add(nn.Dropout(0.5))
	:add(nn.Linear(1000+50,300))
	:add(nn.ReLU())
	:add(nn.Linear(300,2))
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


--[[
local dataSet = {}
dataSet.data = {}
dataSet.data.convData = torch.ones(1000,13):cuda()
dataSet.data.fftData  = torch.ones(1000,128,3):cuda()
dataSet.label = torch.zeros(1000,1):cuda()


setmetatable(dataSet.data,
            {__index = function(t, i)
                            return {t.convData[i],t.fftData[i]}
                        end}
        );

print(dataSet.data.convData[1])
print(dataSet.data.fftData[1])
print(dataSet.data[1])

--print(fourier_extractor:forward(dataSet.data.convData[1]))
--print(tempcon_extractor:forward(dataSet.data.fftData[1]))
--print(net:forward({a,b}))

exit()
--]]
--
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
	trainSet.data = {}
	trainSet.data.convData = CONV_train:cuda()
	trainSet.data.fftData  = FFT_train:cuda() 
	trainSet.label         = trainSet.label:cuda()
	-- Enable access
	setmetatable(trainSet.data,
            {__index = function(t, i)
                            return {t.fftData[i],t.convData[i]}
                        end}
        );
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
	trainer:train(trainSet)
	print('Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')
else
	print('---------------------------------')
        print('Loading already saved model')
	net = torch.load('models/refine_binart')
	if extraIter>0 then
		print('----------------------')
		print('Performing additional '..extraIter..' iterations')
		trainer.maxIteration = extraIter
		trainer:train(trainSet)
	else
		print('No additional iterations...')
	end

end

-- Save the model
torch.save('models/refine_binart', net)

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
testSet.data = {}
testSet.data.convData = CONV_test:cuda()
testSet.data.fftData  = FFT_test:cuda()
testSet.label = testSet.label:cuda()
-- enable access
setmetatable(testSet.data,
             {__index = function(t, i)
                                return {t.fftData[i],t.convData[i]}
			end}
);  
print("Validation data loaded: contains "..testSet:size(1).." samples.")


----------------------------------------------------------------
-- Test the accuracy of the network on the testing data set
----------------------------------------------------------------
print "--------------------------------"
print "Validating artefakt detection..."
print "--------------------------------"
eval.test(testSet,net,true)
