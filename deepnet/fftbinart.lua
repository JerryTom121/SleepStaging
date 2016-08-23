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
nlabels = 3

-----------------------
-- Experiment variables
-----------------------
RETRAIN 	 = false
exp 		 = '5'
aug 		 = '_aug'

--------------------------
-- Architectural variables
--------------------------
max_iterations = 10
extra_iterations = 0
learning_rate  = 0.0001
input_size = 13
layer_1 = 30
layer_2 = 10

------------------------------
-- Neural network architecture
------------------------------
net = nn.Sequential()
-- Add Layer 1:
net:add(nn.Linear(input_size,layer_1))
net:add(nn.ReLU()) 
-- Add Layer 2:
net:add(nn.Linear(layer_1,layer_2))
net:add(nn.ReLU()) 
-- Output
net:add(nn.Linear(layer_2,1))

----------------------------------------
-- Neural network optimization procedure
----------------------------------------
criterion = nn.SoftMarginCriterion()
trainer = nn.StochasticGradient(net,criterion)
trainer.learningRate = learning_rate
trainer.maxIteration = max_iterations
net = net:cuda()
criterion = criterion:cuda()


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


if RETRAIN or extra_iterations>0 then
	-----------------------------------------------------------------
	-- Load training data sets from .CSV files of selected experiment
	-----------------------------------------------------------------
	train_set = inout.load_dataset('../../CSV/train_exp'..exp..aug..'.csv',1,1)
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
	train_set.data  = train_set.data:cuda()
	train_set.label = train_set.label:cuda()
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
	print('---------------------------------')
        print('Loading already saved model')
	net = torch.load('models/fftbinart'..exp)
	if extra_iterations>0 then
		print('----------------------')
		print('Performing additional '..extra_iterations..' iterations')
		trainer = nn.StochasticGradient(net, criterion)
		trainer.learningRate = learning_rate
		trainer.maxIteration = extra_iterations
		trainer:train(train_set)
	else
		print('No additional iterations...')
	end

end


-----------------
-- Save the model
-----------------
torch.save('models/fftbinart'..exp, net)


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
eval.test(test_set,net,false)
