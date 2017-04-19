------------------------------------------------------------------------------
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
cmd:option('-trainedModelPath', '', 'Path to previously trained model')
cmd:option('-inputSize', 3*512, 'The length of the input signal')
cmd:option('-numChannels', 3, 'The number of input channels')
cmd:option('-calibrationPath', '', 'Path to calibration data set')
cmd:text()
params = cmd:parse(arg)

--[[
mlp = nn.Sequential()
mlp:add(nn.CMul(5, 1))

y = torch.rand(5, 4)

mlp:forward
--]]

print("## Load trained model")
trainedModel = torch.load(params.trainedModelPath)

print("## Construct input adaptor")
inputAdaptor = nn.Sequential():add(nn.CMul(1, 1, params.numChannels))

--[[
print("## Construct exteneded model")
local model = nn.Sequential():add(nn.CMul(1, 1, params.numChannels)):add(trainedModel:clone()):cuda()

print("## Loading calibration data...")
local calibset = inout.load_dataset(params.calibrationPath, params.numChannels, 1)

print("## Construct search grid")
s = torch.Tensor(6)
s[1] = 2; s[2] = 1; s[3] = 0.5; s[4] = -0.5; s[5] = -1; s[6] = -2;
print(s)

--[[
for i = 1, s:size(1) do
	for j = 1, s:size(1) do
		for k = 1, s:size(1) do
			model:get(1).weight[1][1][1] = s[i]
			model:get(1).weight[1][1][2] = s[j]
			model:get(1).weight[1][1][3] = s[k]
			print("Weights:")
			print(model:get(1).weight)
			print("Evaluation:")
			eval.predict_and_evaluate(calibset, model)
		end
	end
end
--]]
--[[
model:get(1).weight[1][1][1] = -1
model:get(1).weight[1][1][2] = -2
model:get(1).weight[1][1][3] = -2
eval.predict_and_evaluate(calibset, model)
--]]
--

--inputAdaptor:get(1).weight = inputAdaptor:get(1).weight * 0 - 1
print(inputAdaptor:get(1).weight)

x = torch.rand(1,7,3)
print(x:size())
print(x)
print(inputAdaptor:forward(x))

--
-- Initialize mini-batch trainer
local trainer = nn.IAGD(inputAdaptor, trainedModel, calibset)

print("## Initial Evaluation of the full model on calibration data...")
eval.predict_and_evaluate(calibset, trainer:getModel())

print("## Initial Evaluation of the base model on calibration data...")
eval.predict_and_evaluate(calibset, trainer:getBaseModel())

-- Perform training
for epoch = 1, 5 do
	print("\n## Epoch #"..epoch)
	trainer:train()
	print("## Learned weights:")
	print(trainer:getWeights())
	print("## Evaluating current model on calibration data...")
	eval.predict_and_evaluate(calibset, trainer:getModel())
end

print("## Final Evaluation of the base model on calibration data...")
eval.predict_and_evaluate(calibset, trainer:getBaseModel())
--]]
