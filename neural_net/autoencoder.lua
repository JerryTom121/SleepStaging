require 'nn'
require 'paths'
require 'csvigo'

maxIterations = 4
learningRate  = 0.001
numArtefaktTypes = 4
-- layer 1
L1_fm_in  = 1  -- number of input feature maps
L1_fm_out = 10	  -- number of output feature maps = number of different kernels
L1_kernel = 10 -- kernel size
L1_stride = 1  -- stride length
-- layer 2
L2_fm_in  = L1_fm_out
L2_fm_out = 3
L2_kernel = 10
L2_stride = 1

-- Add Layer 0
signalLenght   = 128
numFeatureMaps = 1
net = nn.Sequential()
-- Add Layer 1
--net:add(nn.TemporalConvolution(L1_fm_in, L1_fm_out, L1_kernel, L1_stride))
--net:add(nn.ReLU()) 
--signalLenght   = (signalLenght-L1_kernel)/L1_stride+1 
--numFeatureMaps = L1_fm_out
-- Add Layer 2
--net:add(nn.TemporalConvolution(L2_fm_in, L2_fm_out, L2_kernel, L2_stride))
--net:add(nn.ReLU())
--signalLenght   = (signalLenght-L2_kernel)/L2_stride+1 
--numFeatureMaps = L2_fm_out
-- Add Layer 3
net:add(nn.View(numFeatureMaps*signalLenght))
--net:add(nn.Linear(numFeatureMaps*signalLenght, 1))
net:add(nn.Linear(numFeatureMaps*signalLenght, numArtefaktTypes))
net:add(nn.ReLU()) 
-- Add Layer 4 
net:add(nn.View(numArtefaktTypes,1))
net:add(nn.TemporalMaxPooling(numArtefaktTypes))


--print(net:forward(torch.rand(128,1)):size()) --[DEBUG]
--print(net:forward(torch.rand(128,1))) --[DEBUG]
print('Minet - autoencoding\n' .. net:__tostring());


-- Load training and testing data sets 
-- from .CSV files of selected experiment
local trainSet = {}
local testSet  = {}
local trainCSV  = torch.Tensor(csvigo.load{path='/home/djordje/Desktop/CSVData/train_exp10.csv',mode='raw'})
local testCSV   = torch.Tensor(csvigo.load{path='/home/djordje/Desktop/CSVData/test_exp10.csv',mode='raw'})
local numFeatures      = trainCSV:size(2)-2
local numTrainSamples  = trainCSV:size(1)
local numTestSamples   = testCSV:size(1)
-- Create data sets by discarding first 
-- column and using the last one as labels
a = trainCSV[{{},{2,trainCSV:size(2)-1}}]
trainSet.data  = trainCSV[{{},{2,trainCSV:size(2)-1}}]
trainSet.label = trainCSV[{{},{trainCSV:size(2)}}]
testSet.data   = testCSV[{{},{2,testCSV:size(2)-1}}]
testSet.label  = testCSV[{{},{testCSV:size(2)}}]
-- Reshape data sets to fit the convolutional 
-- network architecture
trainSet.data = torch.reshape(trainSet.data,numTrainSamples,numFeatures,1)
testSet.data  = torch.reshape(testSet.data,numTrainSamples,numFeatures,1)
-- Some preparation for training of
-- our convolutional neural network
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


-- Setup the tranining parameters for
-- our neural network
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

--[[
print('---------------------------------')
timer = torch.Timer()
trainer:train(trainSet)
print('Time elapsed for training neural net: ' .. timer:time().real .. ' seconds')
print('---------------------------------')
torch.save('models/autoencoder', net)
--]]
net = torch.load('models/minet')


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
    	if prediction[1][1]<0 then
        	hits = hits + 1
        -- not detected
        else
        	fn = fn + 1
        end
    else
    	-- false positive
    	if prediction[1][1]<0 then
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
    	if prediction[1][1]<0 then
        	hits = hits + 1
        -- not detected
        else
        	fn = fn + 1
        end
    else
    	-- false positive
    	if prediction[1][1]<0 then
    		fp = fp + 1
    	end
    end
end

print(hits)
print(fp)
print(fn)
print "------------------------------"