------------------------------------------------------------------------------
-- This script loads the configuration of NN model and the optimization 
-- procedure, and does the training of the model on the specified data set 
--
-- @arg1 str path to training data
-- @arg2 str path to where we save trained model
-------------------------------------------------------------------------------

require 'nn'
require 'cunn'
require 'paths'
require 'sslib.deepnet.lib.optimization'
local inout = require 'sslib.deepnet.lib.inout'
local debug = require 'sslib.deepnet.lib.debug'


training_data_path = arg[1]
trained_model_path = arg[2]

-- Pick a network configuration
local config_file = 'temporal_convolution'

-- Fetch model and optimization parameters
paths.dofile('configs/' .. config_file .. '.lua')
network      = getModel()
optimization = getOptimization()
experiment   = getExperiment()

-- Output parameters
debug.outputParameters(network,optimization,experiment)

-- Load training data
print("## Load training data...")
dataset = inout.load_dataset(training_data_path,3,1)
print (dataset:size())

-- Preform the training procedure
print('## Training of the neural network begins...')
local timer = torch.Timer()
trainer = nn.MBGD(network,nn.ClassNLLCriterion(),optimization,config_file)
trainer:train(dataset)
print('## Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')

-- Save the trained model
torch.save(trained_model_path..experiment.name,network)
