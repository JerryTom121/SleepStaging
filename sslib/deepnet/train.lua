------------------------------------------------------------------------------
-- This script loads specified architecture of NN model and the optimization 
-- procedure parameters, and does the training of the model on the specified 
-- training data set. In each iteration, the model is evaluated on the holdout
-- data set, which is used to decide at which point in the training process our
-- model is the most accurate. To enable parallel training of several models, 
-- in the name of each model, a number of gpu device will be contained.
--
-- @arg1 str path to training data given in .csv format
-- @arg2 str path to holdout data given in .csv format
-- @arg3 str path to where we save final trained model
-- @arg4 str name of the neural network architecture
-- @arg5 str number of gpu on which we train
-------------------------------------------------------------------------------

require 'nn'
require 'cunn'
require 'paths'
require 'math'
require 'sslib.deepnet.lib.optimization'
local inout = require 'sslib.deepnet.lib.inout'
local debug = require 'sslib.deepnet.lib.debug'
local eval  = require 'sslib.deepnet.lib.eval'
local util  = require 'sslib.deepnet.lib.util'

-- Read command line arguments
local trainsetpath = arg[1]
local holdoutpath = arg[2]
local trainedmodelpath = arg[3]
local architecture = arg[4]
local gpu = arg[5]

-- Fetch architecture and optimization parameters of the model
paths.dofile('architecture/' .. architecture .. '.lua')
local model = getModel()
local optimization = getOptimization()
debug.outputParameters(model, optimization)

-- Fetch training data set
print("## Loading training data...")
local trainset = inout.load_dataset(trainsetpath, 3, 1)

-- Temporary solution to speed up learning 
-- to be used only before recurrency is introduced!!!
-- we remove all rows which contain label '4'-artifact
-- and an ambigious scorings - '5'
selected = trainset.label:lt(4):nonzero()[{{}, 1}]
trainset.data = trainset.data:index(1, selected)
trainset.label = trainset.label:index(1, selected)

-- Fetch holdout data set
print("## Loading holdout data...")
holdout = inout.load_dataset(holdoutpath, 3, 1)

-- Further initialization
local trainedmodels = {}
local trainacc = torch.Tensor(optimization.iterations)
local holdoutacc = torch.Tensor(optimization.iterations)

-- Initialize trainer
trainer = nn.MBGD(optimization, model, trainset)

-- Perform training
for iter = 1, optimization.iterations do
	-- Do iteration
	print("\n## Iteration #"..iter)
	trainer:train()
	-- Evaluate current model on holdout data
	print("## Evaluating current model on holdout data...")
	trainacc[iter] = eval.predict_and_evaluate(trainset, trainer:getModel())
	holdoutacc[iter] = eval.predict_and_evaluate(holdout, trainer:getModel())
	-- Report current error
	print('==> Accuracy on training data = '..trainacc[iter]..'; Accuracy on holdout data = '..holdoutacc[iter])
	-- Save current model
	trainedmodels[iter] = trainer:getModel():clone()
end

-- Save model which has highest accuracy on holdout data
local val, ind = holdoutacc:topk(1, true)
torch.save(trainedmodelpath..architecture..'_'..gpu, trainedmodels[ind[1]])
