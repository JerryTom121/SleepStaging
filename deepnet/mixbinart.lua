require 'nn'
require 'paths'
require 'csvigo'
require 'cunn'
require 'gnuplot'
local inout = require 'lib.inout'
local eval  = require 'lib.eval'
local training = require 'lib.SGD'

------------------------------------------
-- Constants: not to be changed in general
------------------------------------------
nchannels 	= 3		
signal_length 	= 128
fft_features    = 13
nlabels 	= 3

-----------------------
-- Experiment variables
-----------------------
RETRAIN 	= true   -- Retrain the network or load existing model
exp 		= '9'    -- Number of experiment
aug 		= '_aug' -- Artifact augmentation type

--------------------------
-- Architectural variables
--------------------------
-- The number of iterations to be performed in case we are retraining the network
max_iterations = 7
-- The number of additional iterations to perform in case we are using the previous network
extra_iterations = 0
-- The learning rate used during the network training
learning_rate  = 0.0002
learning_rate_decay = 0.1
-- Layer 1 (temporal convolution) parameters
l1_feature_maps = 8
l1_kernel_size  = 20
l1_stride_size  = 1
-- Layer 2 (temporal convolution) parameters
l2_feature_maps = 12
l2_kernel_size  = 10
l2_stride_size  = 1

-- Fourier module
----------------
layer_1 = 30
layer_2 = 10

------------------------------
-- Neural network architecture
------------------------------

-- Convolutional module
-----------------------
convnet = nn.Sequential()
-- Add Layer 1:
convnet:add(nn.TemporalConvolution(nchannels,l1_feature_maps,l1_kernel_size,l1_stride_size))
convnet:add(nn.ReLU()) 
signal_length  = (signal_length-l1_kernel_size)/l1_stride_size + 1 
-- Add Layer 2
convnet:add(nn.TemporalConvolution(l1_feature_maps,l2_feature_maps,l2_kernel_size,l2_stride_size))
convnet:add(nn.ReLU())
signal_length   = (signal_length-l2_kernel_size)/l2_stride_size + 1 
-- Add Layer 3: fully connected layer
convnet:add(nn.View(l2_feature_maps*signal_length))
convnet:add(nn.Linear(l2_feature_maps*signal_length, 3))
convnet:add(nn.View(-1))

-- Fourier module
-----------------
fftnet = nn.Sequential()
-- Add Layer 1:
fftnet:add(nn.Linear(fft_features,layer_1))
fftnet:add(nn.ReLU())
-- Add Layer 2:
fftnet:add(nn.Linear(layer_1,layer_2))
fftnet:add(nn.ReLU())
-- Output
fftnet:add(nn.Linear(layer_2,3))
fftnet:add(nn.View(-1))

-- Network
----------
mix = nn.ParallelTable()
	 :add(convnet)
         :add(fftnet)
net = nn.Sequential()
	 :add(mix)
         :add(nn.JoinTable(1))
	 :add(nn.Linear(6,1))
         :add(nn.View(-1))


------------------------------------
---------------- DEBUG -------------
------------------------------------
print("-------------------------")
print("The network architecture:")
print("-------------------------")
print(net:__tostring());
print("------------------------------------")
print("Chosen parameters in this round are:")
print("------------------------------------")
print("The chosen experiment: "..exp)
print("The augmentation method: "..aug)
print("Iterations: "..max_iterations)
print("Learning rate: "..learning_rate)
print("Layer 1 (feature maps,kernel): "..l1_feature_maps,l1_kernel_size)
print("Layer 2 (feature maps,kernel): "..l2_feature_maps,l2_kernel_size)



local train_set_conv = {}
local train_set_fft  = {}


if RETRAIN or extra_iterations>0 then
	-----------------------------------------------------------------
	-- Load training data sets from .CSV files of selected experiment
	-----------------------------------------------------------------

	print("------------------------")
	print("Loading training data...")
	print("------------------------")
	train_set = inout.load_dataset('../../CSV/train_exp'..exp..aug..'.csv',1,1)
	print("Input dimensions:")
	print(train_set.data:size())
	train_set.data = train_set.data:cuda()
	print("Label dimensions:")
	print(train_set.label:size())
	train_set.label = train_set.label:cuda()


	print("------------------------------------------")
	print("Reformating data to fit different modules:")
	print("------------------------------------------")
	train_set_conv = train_set.data[{{},{1,3*128}}]
	train_set_conv = torch.reshape(train_set_conv,train_set.data:size(1),3,128)
        train_set_conv = train_set_conv:transpose(2,3)
	train_set_fft  = train_set.data[{{},{3*128+1,3*128+13}}]

	print("Conv module input dimensions:")
	print(train_set_conv:size())
	train_set_conv = train_set_conv:cuda()

	print("FFT module input dimensions:")
	print(train_set_fft:size())
	train_set_fft  = train_set_fft:cuda()

end


----------------------------------------
-- Neural network optimization procedure
----------------------------------------
criterion = nn.SoftMarginCriterion()
trainer = nn.SGD(net,criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = max_iterations
trainer.learningRateDecay = learning_rate_decay
net = net:cuda()
criterion = criterion:cuda()

---------------------------------------------------
-- Train the network or simply load it from the file
----------------------------------------------------
if RETRAIN then
	print('---------------------------------')
	print('Training of the neural network begins...')
	collectgarbage()
    	timer = torch.Timer()
	trainer:train(train_set_conv,train_set_fft,train_set.label)
	print('Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')
else
	print('---------------------------------')
        print('Loading already saved model')
	net = torch.load('models/binart'..exp)
	if extra_iterations>0 then
		print('----------------------')
		print('Performing additional '..extra_iterations..' iterations')
		trainer = nn.SGD(net, criterion)
		trainer.learningRate = learning_rate
		trainer.maxIteration = extra_iterations
		trainer.learningRateDecay = learning_rate_decay
		trainer:train(train_set_conv,train_set_fft,train_set.label)
	else
		print('No additional iterations...')
	end

end


-----------------
-- Save the model
-----------------
torch.save('models/binart'..exp, net)


----------------------------------------------------------------
-- Load testing data sets from .CSV files of selected experiment
----------------------------------------------------------------
test_set  = inout.load_dataset('../../CSV/test_exp'..exp..'.csv',1,nlabels)
------------------------------------
-------------- DEBUG ---------------
------------------------------------
print("---------------------")
print("Testing data loaded:")
print("---------------------")
print("Input dimensions:")
print(test_set.data:size())
print("Label dimensions:")
print(test_set.label:size())
test_set.data  = test_set.data:cuda()
test_set.label = test_set.label:cuda()



-----------------------------------------------------------
-- Test the accuracy of the network on the testing data set
-----------------------------------------------------------
print "------------------------------"
print "Testing set artefakt detection"
eval.test(test_set,net)
