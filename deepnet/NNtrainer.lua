---------------------------------------------------------------------------
-- This script loads the configuration of NN model and the optimization 
-- procedure, and does the training of the model on the specified data set 
--
-- @param => the configuration of the model/optimization procedure/data set 
---------------------------------------------------------------------------

require 'nn'
require 'cunn'
require 'paths'
require 'lib.MBGD'
local inout = require 'lib.inout'
local debug = require 'lib.debug'

------------------------------
-- Parse command line argument
------------------------------
local configFile = arg[1]
assert(configFile=="spectral" or configFile=="temporal" or configFile=="hybrid","ERROR: No existing model is selected!")

----------------------------
-- Fetch training parameters
----------------------------
paths.dofile('configs/' .. configFile .. '.lua')
network      = getModel()
optimization = getOptimization()
experiment   = getExperiment()

-----------------------
-- Print the parameters
-----------------------
debug.outputParameters(network,optimization,experiment)

---------------------
-- Load training data
---------------------
print("## Load training data...")
if configFile=="hybrid" then
	trainSet = inout.load_dataset('../../CSV/train_exp'..experiment.number..experiment.augType..'.csv',1,1)
	-- split into temporal and fft matrix
	CONV_train = torch.reshape(trainSet.data[{{},{1,experiment.numChan*experiment.epochSize}}],
				   trainSet.data:size(1),experiment.numChan,experiment.epochSize):transpose(2,3)
	FFT_train  = trainSet.data[{{},{experiment.numChan*experiment.epochSize+1,experiment.numChan*experiment.epochSize+experiment.numFFTfeat}}]
	-- print training set dimensions 
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
else
        trainSet = inout.load_dataset('../../CSV/train_exp'..experiment.number..experiment.augType..'.csv',experiment.numChan,1)
        print("## Input dimensions: "..trainSet.data:size(1).." x "..trainSet.data:size(2))
end

-------------------------------------
-- Load pretrained model if specified
-------------------------------------
if optimization.pretrained then
        print('## Loading already saved model..')
        network = torch.load('models/'..configFile)
end

---------------------------------
-- Preform the training procedure
---------------------------------
print('## Training of the neural network begins...')
local timer = torch.Timer()
trainer = nn.MBGD(network,nn.ClassNLLCriterion(),optimization)
trainer:train(trainSet)
print('## Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')

-------------------------
-- Save the trained model
-------------------------
torch.save('models/'..configFile..'_extractor',network:get(1))
torch.save('models/'..configFile,network)
