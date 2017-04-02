------------------------------------------------------------------------------
-- The script first loads specified architecture of NN and then does the 
-- training of the model on specified training data set. In each epoch, the
-- model is evaluated on the holdout data set, which is used to decide when to 
-- stop the training process (early stopping procedure). To enable parallel 
-- training of several models on different GPUs in the name of each model a 
-- number of GPU device will be contained. Necessary parameters are passed
-- through command line.
-------------------------------------------------------------------------------

require 'torch'
require 'nn'
require 'cunn'
require 'paths'
require 'math'
require 'sslib.deepnet.lib.optimization'
local inout = require 'sslib.deepnet.lib.inout'
local eval  = require 'sslib.deepnet.lib.eval'
local util  = require 'sslib.deepnet.lib.util'

-- Read and parse command line arguments
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training starts..')
cmd:text()
cmd:text('Options')
cmd:option('-learningRate', 0.0005, 'Learning rate')
cmd:option('-learningRateDecay', 0.01, 'Learning rate decay')
cmd:option('-momentum', 0.5, 'Nesterov momentum for SGD')
cmd:option('-weightDecay', 0.001, 'Weight decay regularization')
cmd:option('-dropout', 0.5, 'Dropout regularization')
cmd:option('-batchSize', 1, 'The size of a mini-batch')
cmd:option('-maxEpochs', 70, 'Max number of epochs/iterations')
cmd:option('-nclasses', 3, 'Number of classes depends on the problem solved')
cmd:option('-trainPath', '', 'Path to training data')
cmd:option('-holdoutPath', '', 'Path to holdout data')
cmd:option('-architecture', '', 'The name of the NN architecture to be loaded')
cmd:option('-gpu', 0, 'The number of gpu to run on')
cmd:option('-inputSize', 3*512, 'The length of the input signal')
cmd:option('-numChannels', 3, 'The number of input channels')
cmd:text()
params = cmd:parse(arg)
cmd:log('logs/gpu' .. params.gpu, params) -- make sure to log events

-- Load and output model architecture
print("## NN architecture:")
paths.dofile('architecture/'..params.architecture..'.lua')
print(model)

-- Fetch training data set
print("## Loading training data...")
local trainset = inout.load_dataset(params.trainPath, params.numChannels, 1)

-- Fetch holdout data set
print("## Loading holdout data...")
local holdout = inout.load_dataset(params.holdoutPath, params.numChannels, 1)

-- Further initialization
local trainedmodels = {}
local trainacc = torch.Tensor(params.maxEpochs)
local holdoutacc = torch.Tensor(params.maxEpochs)
local trainer = nn.MBGD(params, model, trainset)

-- Perform training
for epoch = 1, params.maxEpochs do
	print("\n## Epoch #"..epoch)
	trainer:train()
	print("## Evaluating current model on training and holdout data...")
	trainacc[epoch] = eval.predict_and_evaluate(trainset, trainer:getModel())
	holdoutacc[epoch] = eval.predict_and_evaluate(holdout, trainer:getModel())
	print('## Accuracy on training data = '..trainacc[epoch]) 
	print('## Accuracy on holdout data = '..holdoutacc[epoch])
	trainedmodels[epoch] = trainer:getModel():clone()
	-- decrease learning rate
	trainer:setLearningRate(params.learningRate/(1+epoch*params.learningRateDecay))
end

-- Save model which has highest accuracy on holdout data
local val, ind = holdoutacc:topk(1, true)
torch.save('models/'..params.architecture..'_'..params.gpu, trainedmodels[ind[1]])
