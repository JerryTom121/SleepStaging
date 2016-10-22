require 'nn'
require 'paths'
require 'csvigo'
require 'cunn'
require 'gnuplot'
local inout = require 'lib.inout'
local eval  = require 'lib.eval'

------------------------------------------
-- Constants: not to be changed in general
------------------------------------------
-- use CUDA
CUDA = true
-- use all 3 channels
nchannels = 3		
-- epoch length
signal_length = 128
--signal_length = 256
-- number of labels on the test data;  apart from 1/0 label which indicates whether 
-- the sample is artifact, we have some extra information such as e.g. neighbour labels
nlabels = 3

-----------------------
-- Experiment variables
-----------------------
-- Specify whether we should retrain the model or use the previous one
RETRAIN 	 = true
-- The number of the experiment to be performed
exp 		 = '6'
-- The augmentation type of the selected experiment
aug 		 = '_rot_mir'

--------------------------
-- Architectural variables
--------------------------
-- The number of iterations to be performed in case we are retraining the network
max_iterations = 6
-- The number of additional iterations to perform in case we are using the previous network
extra_iterations = 0
-- The learning rate used during the network training
learning_rate  = 0.0002
-- Layer 1 (temporal convolution) parameters
l1_feature_maps = 8
l1_kernel_size  = 20
l1_stride_size  = 1
-- Layer 2 (temporal convolution) parameters
l2_feature_maps = 12
l2_kernel_size  = 10
l2_stride_size   = 1

------------------------------
-- Neural network architecture
------------------------------
net = nn.Sequential()
-- Add Layer 1:
net:add(nn.TemporalConvolution(nchannels,l1_feature_maps,l1_kernel_size,l1_stride_size))
net:add(nn.ReLU()) 
signal_length  = (signal_length-l1_kernel_size)/l1_stride_size + 1 
-- [Add Layer 1.1]
-- Max Pooling : Added Additionally
-- net:add(nn.TemporalMaxPooling(2,2))
-- signalLenght = (signalLenght-2)/2+1 
-- Add Layer 2
net:add(nn.TemporalConvolution(l1_feature_maps,l2_feature_maps,l2_kernel_size,l2_stride_size))
net:add(nn.ReLU())
signal_length   = (signal_length-l2_kernel_size)/l2_stride_size + 1 
-- [Add Layer 2.1]
-- Max Pooling : Added Additionally
-- net:add(nn.TemporalMaxPooling(2,2))
-- signalLenght = (signalLenght-2)/2+1 
-- Add Layer 3: fully connected layer
net:add(nn.View(l2_feature_maps*signal_length))
net:add(nn.Linear(l2_feature_maps*signal_length, 1))

----------------------------------------
-- Neural network optimization procedure
----------------------------------------
criterion = nn.SoftMarginCriterion()
trainer = nn.StochasticGradient(net,criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = max_iterations
if CUDA then
	net = net:cuda()
	criterion = criterion:cuda()
end


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


if RETRAIN or extra_iterations>0 then
	-----------------------------------------------------------------
	-- Load training data sets from .CSV files of selected experiment
	-----------------------------------------------------------------
	train_set = inout.load_dataset('../../CSV/train_exp'..exp..aug..'.csv',nchannels,1)
	------------------------------------
	-------------- DEBUG ---------------
	------------------------------------
	print("---------------------")
	print("Training data loaded:")
	print("---------------------")
	print("Input dimensions:")
	print(train_set.data:size())
	print("Label dimensions:")
	print(train_set.label:size())
	if CUDA then
		train_set.data  = train_set.data:cuda()
		train_set.label = train_set.label:cuda()
	end
end


---------------------------------------------------
-- Train the network or simply load it from the file
----------------------------------------------------
if RETRAIN then
	print('---------------------------------')
	print('Training of the neural network begins...')
	collectgarbage()
    	timer = torch.Timer()
	trainer:train(train_set)
	print('Time elapsed for training neural net: ' .. timer:time().real .. ' seconds\n')
else
	print('-----------------------------')
        print('Loading already saved model..')
	net = torch.load('models/binart'..exp)
	if extra_iterations>0 then
		print('----------------------')
		print('Performing additional '..extra_iterations..' iterations')
		trainer = nn.StochasticGradient(net, criterion)
		trainer.learningRate = learning_rate
		trainer.maxIteration = extra_iterations
		trainer:train(train_set)
	else
		print('(no additional iterations)')
	end

end


-----------------
-- Save the model
-----------------
torch.save('models/binart'..exp, net)


----------------------------------------------------------------
-- Load testing data sets from .CSV files of selected experiment
----------------------------------------------------------------
test_set  = inout.load_dataset('../../CSV/test_exp'..exp..'.csv',nchannels,nlabels)
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
if CUDA then
	test_set.data  = test_set.data:cuda()
	test_set.label = test_set.label:cuda()
end



-----------------------------------------------------------
-- Test the accuracy of the network on the testing data set
-----------------------------------------------------------
print "------------------------------"
print "Testing set artefakt detection"
eval.test(test_set,net,false)



-------------------------------------------------
-- Test the accuracy of the combination of models
-------------------------------------------------
print "------------------------------------------------"
print "Testing set artefakt detection of a combination:"
fourier_net = torch.load('models/fftbinart5')
fourier_test_set  = inout.load_dataset('../../CSV/test_exp5.csv',1,nlabels)
fourier_test_set.data  = fourier_test_set.data:cuda()
fourier_test_set.label = fourier_test_set.label:cuda()
eval.test2(test_set,net,fourier_test_set,fourier_net)

