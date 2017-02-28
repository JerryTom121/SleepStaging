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
architecture = arg[3]

paths.dofile('architecture/' .. architecture .. '.lua')
network      = getModel()
optimization = getOptimization()
experiment   = getExperiment()

debug.outputParameters(network, optimization, experiment)

print("## Load training data...")
dataset = inout.load_dataset(training_data_path, 3, 1)

print("## Class weights: ")
class_weights = util.get_class_weights(dataset, optimization.classes)
print(class_weights)

print('## Training of the neural network begins...')
local timer = torch.Timer()
trainer = nn.MBGD(network, nn.ClassNLLCriterion(class_weights), optimization, architecture, optimization.classes)
trainer:train(dataset)

print('## Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')
torch.save(trained_model_path .. architecture, network)
