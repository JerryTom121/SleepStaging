------------------------------------------------------------------------------
-- This script loads the configuration of NN model and the optimization 
-- procedure, and does the training of the model on the specified data set 
--
-- @arg1 str path to training data
-- @arg2 str path to where we save trained model
-- @arg3 the name of the neural network architecture
-------------------------------------------------------------------------------

require 'nn'
require 'cunn'
require 'paths'
require 'math'
require 'sslib.deepnet.lib.optimization'
local inout = require 'sslib.deepnet.lib.inout'
local debug = require 'sslib.deepnet.lib.debug'
local util = require 'sslib.deepnet.lib.util'


training_data_path = arg[1]
trained_model_path = arg[2]
config_file = arg[3]

paths.dofile('configs/' .. config_file .. '.lua')
network      = getModel()
optimization = getOptimization()
experiment   = getExperiment()

debug.outputParameters(network,optimization,experiment)

print("## Load training data...")
dataset = inout.load_dataset(training_data_path,3,1)

class_weights = util.get_class_weights(dataset)
print("## Class weights: ")
print(class_weights)

print('## Training of the neural network begins...')
local timer = torch.Timer()
trainer = nn.MBGD(network,nn.ClassNLLCriterion(class_weights),optimization,config_file)
trainer:train(dataset)
print('## Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')

torch.save(trained_model_path..config_file,network)
