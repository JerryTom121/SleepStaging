require 'nn'
require 'paths'
require 'csvigo'
require 'cunn'
require 'gnuplot'



-- Useful functions:
----------------------------------------------------------------------
function string:split(sep)
  --------------------------------------------
  --- Split string using given separator <sep>
  -------------------------------------------
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end

function readCSV(filePath)
	-------------------------------
	-- Read data from CSV to tensor
	-------------------------------
	-- Read given file
	local csvFile = io.open(filePath, 'r')  
	-- Count number of rows and columns in file
	local COLS
	local ROWS
	local i = 0
	for line in io.lines(filePath) do
	  if i == 0 then
	    COLS = #line:split(',')
	  end
	  i = i + 1
	end
	ROWS = i 
	-- Load file into Tensor
	local data = torch.Tensor(ROWS, COLS)
	local i = 0  
	for line in csvFile:lines('*l') do  
	  i = i + 1
	  local l = line:split(',')
	  for key, val in ipairs(l) do
	    data[i][key] = val
	  end
	end
	-- Close file
	csvFile:close() 
	-- Return tensor
	return data
end


-- User defined variables
-------------------------
RETRAIN = false
VISUALIZE = false
CUDA    = true
exp = '7'
nlabels = 4 -- 4 for intersection data
exp_num_channels = 3
aug = '_rot_mir'
--aug = '_aug'
signalLenght  = 128 --for exp1,exp2,exp4
--signalLenght = 512 --for ??
--signalLenght = 65 -- for exp3
maxIterations = 10
extraIterations = 1 -- used only in case we do not retrain the model from start
learningRate  = 0.0002


collectgarbage()

-- Architecture
------------------------------------------------
-- layer 1
L1_fm_in  = exp_num_channels  -- number of input feature maps
L1_fm_out = 8 -- number of output feature maps = number of different kernels
L1_kernel = 20 -- kernel size
L1_stride = 1  -- stride length
-- layer 2
L2_fm_in  = L1_fm_out
L2_fm_out = 12
L2_kernel = 10
L2_stride = 1


-- Add Layer 0
net = nn.Sequential()
-- Add Layer 1
L1 = nn.TemporalConvolution(L1_fm_in, L1_fm_out, L1_kernel, L1_stride)
net:add(L1)
net:add(nn.ReLU()) 
signalLenght   = (signalLenght-L1_kernel)/L1_stride+1 
numFeatureMaps = L1_fm_out

--- Max Pooling : Added Additionally
--net:add(nn.TemporalMaxPooling(2,2))
--signalLenght = (signalLenght-2)/2+1 


-- Add Layer 2
net:add(nn.TemporalConvolution(L2_fm_in, L2_fm_out, L2_kernel, L2_stride))
net:add(nn.ReLU())
signalLenght   = (signalLenght-L2_kernel)/L2_stride+1 
numFeatureMaps = L2_fm_out

-- Add Layer 3
--net:add(nn.TemporalConvolution(L2_fm_out, 8, 3, 1))
--net:add(nn.ReLU())
--signalLenght   = (signalLenght-3)/1+1 
--numFeatureMaps = 8

--- Max Pooling : Added Additionally
--net:add(nn.TemporalMaxPooling(2,2))
--signalLenght = (signalLenght-2)/2+1 

print("Number of channels is:"..L1_fm_in)
print("Length of signal going in the fully connected layer: "..signalLenght)
-- Add Layer 3
net:add(nn.View(numFeatureMaps*signalLenght))
net:add(nn.Linear(numFeatureMaps*signalLenght, 1))
-- Print out
print(net:__tostring());
print("Chosen parameters in this round are:")
print("-----------------------------------")
print("The experiment: "..exp)
print("The augmentation method: "..aug)
print("Iterations: "..maxIterations)
print("Learning rate: "..learningRate)
print("Layer1(fm,kernel): "..L1_fm_out,L1_kernel)
print("Layer2(fm,kernel): "..L2_fm_out,L2_kernel)

-- Setup the tranining parameters for our neural network
--------------------------------------------------------
criterion = nn.SoftMarginCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = learningRate
trainer.maxIteration = maxIterations


collectgarbage()
-- Load training and testing data sets from .CSV files of selected experiment
-----------------------------------------------------------------------------
local trainSet = {}
local testSet  = {}
--local trainCSV  = torch.Tensor(csvigo.load{path='./CSV/train_exp'..exp..aug..'.csv',mode='raw'})
--local testCSV   = torch.Tensor(csvigo.load{path='./CSV/test_exp'..exp..'.csv',mode='raw'})
local trainCSV  = readCSV('./CSV/train_exp'..exp..aug..'.csv')
local testCSV   = readCSV('./CSV/test_exp'..exp..'.csv')

local numFeatures      = trainCSV:size(2)-2
local numTrainSamples  = trainCSV:size(1)
local numTestSamples   = testCSV:size(1)
-- Create data sets by discarding first column and using the last one as labels
trainSet.data  = trainCSV[{{},{2,trainCSV:size(2)-1}}]
trainSet.label = trainCSV[{{},{trainCSV:size(2)}}]
testSet.data   =  testCSV[{{},{2,testCSV:size(2)-nlabels}}]
testSet.label  =  testCSV[{{},{testCSV:size(2)-nlabels+1,testCSV:size(2)}}]
-- Reshape data sets to fit the convolutional  network architecture
trainSet.data = torch.reshape(trainSet.data,numTrainSamples,exp_num_channels,numFeatures/exp_num_channels)
testSet.data  = torch.reshape(testSet.data,numTestSamples,exp_num_channels,numFeatures/exp_num_channels)

trainSet.data = trainSet.data:transpose(2,3)
testSet.data = testSet.data:transpose(2,3)

--trainSet.data = trainSet.data[{{1,40000},{},{}}]

print("------------------------------------------")
print(trainSet.data:size())
print(trainSet.label:size())
print(testSet.data:size())
print(testSet.label:size())

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
setmetatable(testSet, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
function testSet:size() 
    return self.data:size(1) 
end



collectgarbage()
-- Move to CUDA if specified
----------------------------
if CUDA then
	net = net:cuda()
	criterion = criterion:cuda()
	trainSet.data = trainSet.data:cuda()
	trainSet.label = trainSet.label:cuda()
	testSet.data  = testSet.data:cuda()
	testSet.label = testSet.label:cuda()
end


-- Train the network or simply load it from the file
----------------------------------------------------
if RETRAIN then
	print('---------------------------------')
	print('Training of the neural network begins...')
    	timer = torch.Timer()
	trainer:train(trainSet)
	print('Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')
else
	print('---------------------------------')
        print('Loading already saved model')
	net = torch.load('models/binart')
	if extraIterations>0 then
		print('----------------------')
		print('Performing additional '..extraIterations..' iterations')
		trainer = nn.StochasticGradient(net, criterion)
		trainer.learningRate = learningRate
		trainer.maxIteration = extraIterations
		trainer:train(trainSet)
	else
		print('No additional iterations...')
	end

end
torch.save('models/binart', net)


-- Auxilary function
--------------------
function test(dataset)
    local hits = 0
    local fp = 0
    local fn = 0

    local fp_doubtful  = 0
    local fn_neighbour = 0

    for i=1,dataset:size() do
        local groundtruth = dataset.label[i][1]
        local prediction = net:forward(dataset.data[i])
       
        -- artefakt
        if groundtruth==-1 then
            -- detected
            if prediction[1]<0 then
                hits = hits + 1
            -- not detected
            else
                fn = fn + 1
		-- Check if the missed artefakt is in 
		-- the neighbourhood of another artefakt
		if dataset.label[i][4]==-2 then
			fn_neighbour = fn_neighbour + 1
		end 
            end
        else
            -- false positive
            if prediction[1]<0 then
                fp = fp + 1
		-- check if this false positive has been classified as
		-- an artifact at least by one of the raters
		if dataset.label[i][2]==-1 or dataset.label[i][3]==-1 then
			fp_doubtful = fp_doubtful + 1
		end
            end
        end
    end
    print('Number of guessed artefakts: '..hits)
    print('Number of false positives:   '..fp)
    print('Number of missed artefakts:  '..fn)
    print('----------------------------------------')
    print('Number of doubtful false positives: '..fp_doubtful)
    print('Number of misclassified artifacts which are in the neighbourhood of another: '..fn_neighbour)
end

function save(dataset)
   -- local t = torch.Tensor(dataset:size())
    local file = io.open ("proba2","w")
    local t
    io.output(file)
    for i=1,dataset:size() do
   	local prediction = net:forward(dataset.data[i])
        if prediction[1]<0 then
		t = -1
        else
		t = 1
        end
	io.write(t..'\n')
    end
--    torch.save('proba',t,ascii)
    io.close(file)
end


-- Test the accuracy of the network on the training data set
------------------------------------------------------------
--print "------------------------------"
--print "Training set artefakt detection"
--test(trainSet)

-- Test the accuracy of the network on the testing data set
------------------------------------------------------------
print "------------------------------"
print "Testing set artefakt detection"
test(testSet)



-- Save predictions
-- save(testSet)

-- L1 kernel visualization
--------------------------
if VISUALIZE then
	print("--------------------------------------")
	print("Visualizing weights of the first layer")
	for i=1,L1_fm_out do
		for j=0,exp_num_channels-1 do
			local sig = net:get(1).weight[{i,{j*L1_kernel+1,j*L1_kernel+L1_kernel}}]
			name = 'kernel'..i..'x'..(j+1)
			gnuplot.pngfigure('./images/'..name..'.png')
			gnuplot.title(name)
			gnuplot.axis({'','',-1,1})
			gnuplot.plot(sig)
	                gnuplot.plotflush()
		end
	end
end


