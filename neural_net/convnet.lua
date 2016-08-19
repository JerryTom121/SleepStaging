require 'nn'
require 'paths'
require 'csvigo'


--[[
-- Define Convolutional Neural Network
-- for 0/1 classification
net = nn.Sequential()
-- layer 1
net:add(nn.SpatialConvolution(1, 3, 1, 4)) -- 1 input image channels, 3 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                         -- non-linearity 
net:add(nn.Dropout(0.4))
-- layer 2
net:add(nn.SpatialConvolution(3, 6, 1, 8))
net:add(nn.SpatialMaxPooling(1,2))
net:add(nn.ReLU())               
net:add(nn.Dropout(0.4))
-- layer 3
net:add(nn.SpatialConvolution(6, 12, 1, 16))
net:add(nn.ReLU())               
net:add(nn.Dropout(0.4))
-- layer 4
net:add(nn.View(12*64))
net:add(nn.Linear(12*64,100)) 
net:add(nn.Dropout(0.4))
-- layer 5
net:add(nn.LogSoftMax())
--]]

DEBUG = false
maxIterations = 4
learningRate  = 0.001
-- layer 1
L1_fm_in  = 1  -- number of input feature maps
L1_fm_out = 10  -- number of output feature maps = number of different kernels
L1_kernel = 10 -- kernel size
L1_stride = 1  -- stride length
-- layer 2
L2_fm_in  = L1_fm_out
L2_fm_out = 1
L2_kernel = 10
L2_stride = 1

-- Add Layer 0
signalLenght   = 128
numFeatureMaps = 1
net = nn.Sequential()
-- Add Layer 1
net:add(nn.TemporalConvolution(L1_fm_in, L1_fm_out, L1_kernel, L1_stride))
net:add(nn.ReLU()) 
signalLenght   = (signalLenght-L1_kernel)/L1_stride+1 
numFeatureMaps = L1_fm_out
-- Add Layer 2
net:add(nn.TemporalConvolution(L2_fm_in, L2_fm_out, L2_kernel, L2_stride))
net:add(nn.ReLU())
signalLenght   = (signalLenght-L2_kernel)/L2_stride+1 
numFeatureMaps = L2_fm_out
-- Add Layer 3
net:add(nn.View(numFeatureMaps*signalLenght))
net:add(nn.Linear(numFeatureMaps*signalLenght, 1))


--print(net:forward(torch.rand(128,1)):size()) --[DEBUG]
--print(net:forward(torch.rand(128,1))) --[DEBUG]
print('Minet\n' .. net:__tostring());








-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.outputFrameSize
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  local function scaleDown(name)
    for k,v in pairs(net:findModules(name)) do
      v.weight = v.weight/100
      v.bias   = v.bias/1000
    end
  end
  init 'nn.TemporalConvolution'
  --scaleDown 'nn.TemporalConvolution'
  --scaleDown 'nn.Linear'
  --print(net.modules[1].weight)
  --print(net.modules[4].weight)
end

MSRinit(net)


















-- Load training and testing data sets from .CSV files of selected experiment
-----------------------------------------------------------------------------
local trainSet = {}
local testSet  = {}
local trainCSV  = torch.Tensor(csvigo.load{path='/home/djordje/Desktop/CSVData/train_exp10.csv',mode='raw'})
local testCSV   = torch.Tensor(csvigo.load{path='/home/djordje/Desktop/CSVData/test_exp10.csv',mode='raw'})
local numFeatures      = trainCSV:size(2)-2
local numTrainSamples  = trainCSV:size(1)
local numTestSamples   = testCSV:size(1)
-- Create data sets by discarding first column and using the last one as labels
trainSet.data  = trainCSV[{{},{2,trainCSV:size(2)-1}}]
trainSet.label = trainCSV[{{},{trainCSV:size(2)}}]
testSet.data   = testCSV[{{},{2,testCSV:size(2)-1}}]
testSet.label  = testCSV[{{},{testCSV:size(2)}}]
-- Reshape data sets to fit the convolutional  network architecture
trainSet.data = torch.reshape(trainSet.data,numTrainSamples,numFeatures,1)
testSet.data  = torch.reshape(testSet.data,numTestSamples,numFeatures,1)
-- Some other preparation for training of our convolutional neural network
trainSet.data = trainSet.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
testSet.data  = testSet.data:double()  -- convert from Byte tensor to Double tensor
setmetatable(trainSet, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
function trainSet:size() 
    return self.data:size(1) 
end



-- Setup the tranining parameters for our neural network
--------------------------------------------------------
criterion = nn.SoftMarginCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = learningRate
trainer.maxIteration = maxIterations -- just do 5 epochs of training
-- Some
-- Debugging
--[[
input  = trainSet[2][1]
tarqget = trainSet[2][2]
output =  net:forward(input) -- [DEBUG]
print(output)
print(target)
print(criterion:forward(output,target))
upd = criterion:updateGradInput(output,target)
print(upd)
net:updateGradInput(input,upd)
print(trainSet:size())
print(net:accUpdateGradParameters(input, criterion.gradInput, 0.001))

input  = trainSet[3][1]
target = trainSet[3][2]
output =  net:forward(input) -- [DEBUG]
print(output)
print(target)
print(criterion:forward(output,target))
upd = criterion:updateGradInput(output,target)
print(upd)
net:updateGradInput(input,upd)
print(trainSet:size())
print(net:accUpdateGradParameters(input, criterion.gradInput, 0.001))
--]]


print('---------------------------------')
timer = torch.Timer()
if DEBUG then

	dataset = trainSet
	local iteration = 1
	local currentLearningRate = trainer.learningRate
	local mod = trainer.module
	local criterion = trainer.criterion

	local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
	if not trainer.shuffleIndices then
	  for t = 1,dataset:size() do
	     shuffledIndices[t] = t
	  end
	end


	print("# StochasticGradient: training")

	while true do
	  local currentError = 0
	  for t = 1,dataset:size() do

	     local example = dataset[shuffledIndices[t]]
	--]]

	     local input = example[1]
	     local target = example[2]

	     mod:forward(input)
	     --mod.output = torch.Tensor(1):double()
	     --mod.output[1] = 100

	     print("step = " .. t)
	     print(mod.modules[1].weight)
	     print(mod.modules[4].weight)
	     print("output = ")
	     print(mod.output)
	     print("target = ")
	     print(target)
	     print("criterion_forward = " .. criterion:forward(mod.output, target))
	     currentError = currentError + criterion:forward(mod.output, target)

	     mod:updateGradInput(input, criterion:updateGradInput(mod.output, target))
	     mod:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)

	     if trainer.hookExample then
	        trainer.hookExample(trainer, example)
	     end

	     print("error = " .. currentError)
	     print("####################################")

	     if t>1 then
	     	break
	     end

	  end

	  currentError = currentError / dataset:size()

	  if trainer.hookIteration then
	     trainer.hookIteration(trainer, iteration, currentError)
	  end

	  if trainer.verbose then
	     print("# current error = " .. currentError)
	  end
	  iteration = iteration + 1
	  currentLearningRate = trainer.learningRate/(1+iteration*trainer.learningRateDecay)
	  if trainer.maxIteration > 0 and iteration > trainer.maxIteration then
	     print("# StochasticGradient: you have reached the maximum number of iterations")
	     print("# training error = " .. currentError)
	     break
	  end
	end
else
	trainer:train(trainSet)
end
print('Time elapsed for training neural net: ' .. timer:time().real .. ' seconds')
print('---------------------------------')
torch.save('models/minet', net)


print "------------------------------"
print "Training set artefakt detection"
local hits = 0
local fp = 0
local fn = 0
for i=1,trainCSV:size(1) do
    local groundtruth = trainSet.label[i]
    local prediction = net:forward(trainSet.data[i])
   
    -- artefakt
    if groundtruth[1]==-1 then
    	-- detected
    	if prediction[1]<0 then
        	hits = hits + 1
        -- not detected
        else
        	fn = fn + 1
        end
    else
    	-- false positive
    	if prediction[1]<0 then
    		fp = fp + 1
    	end
    end
end

print(hits)
print(fp)
print(fn)
print "------------------------------"

print "------------------------------"
print "Testing set artefakt detection"
local hits = 0
local fp = 0
local fn = 0
for i=1,trainCSV:size(1) do
    local groundtruth = testSet.label[i]
    local prediction = net:forward(testSet.data[i])
   
    -- artefakt
    if groundtruth[1]==-1 then
    	-- detected
    	if prediction[1]<0 then
        	hits = hits + 1
        -- not detected
        else
        	fn = fn + 1
        end
    else
    	-- false positive
    	if prediction[1]<0 then
    		fp = fp + 1
    	end
    end
end

print(hits)
print(fp)
print(fn)
print "------------------------------"