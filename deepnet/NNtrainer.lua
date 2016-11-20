---------------------------------------------------------------------------
-- This script loads the configuration of NN model and the optimization 
-- procedure, and does the training of the model on the specified data set 
--
-- @param => the configuration of the model/optimization procedure/data set 
---------------------------------------------------------------------------

require 'nn'
require 'cunn'
require 'paths'
require 'lib.optimization'
local inout = require 'lib.inout'
local debug = require 'lib.debug'

------------------------------
-- Parse command line argument
------------------------------
local configFile = arg[1]

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
if experiment.number==0 then
	trainSet = {}
	trainSet.data = {}
	temp = inout.load_dataset('../../CSV/train_exp12_rot.csv',3,1)
	fft  = inout.load_dataset('../../CSV/train_exp14_aug.csv',1,1)
	trainSet.data.temp = temp.data:cuda()
	trainSet.data.fft  = fft.data:cuda()
	trainSet.label     = temp.label:cuda()
	setmetatable(trainSet.data,
                     {__index = function(t, i)
                                     return {t.temp[i],t.fft[i]}
                                end}
        );
	function trainSet:size()
		return trainSet.label:size(1)
	end
else
        trainSet = inout.load_dataset('../../CSV/train_exp'..experiment.number..experiment.augType..'.csv',experiment.numChan,1)
        print("## Input dimensions: ");print(trainSet.data:size())
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
trainer = nn.MBGD(network,nn.ClassNLLCriterion(),optimization,configFile)
trainer:train(trainSet)
print('## Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')

-------------------------
-- Save the trained model
-------------------------
torch.save('models/'..configFile..'_extractor',network:get(1))
torch.save('models/'..configFile,network)
