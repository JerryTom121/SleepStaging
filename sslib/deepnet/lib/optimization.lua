require 'torch'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
local util = require 'sslib.deepnet.lib.util'

local MBGD = torch.class('nn.MBGD')

function MBGD:__init(params, model, dataset)
	-- optimization pars
	self.learningRate = params.learningRate or 0.001
	self.batchSize = params.batchSize or 1
	self.weightDecay = params.weightDecay or 0
	self.momentum = params.momentum or 0
	self.nclasses = params.nclasses
	-- model
	self.model = model
	-- data
	self.dataset = dataset
	-- criterion (class-balanced)
	class_weights = util.get_class_weights(self.dataset, self.nclasses)
	self.criterion = nn.ClassNLLCriterion(class_weights)
end



function MBGD:train()

	-- Initalization
	local criterion = self.criterion
	local batchSize = self.batchSize
	local model = self.model
	local nclasses = self.nclasses
	local dataset = self.dataset
	local optimConfig = {
		learningRate = self.learningRate,
		weightDecay = self.weightDecay,
		momentum = self.momentum
	}

	print("## Learning rate is "..optimConfig.learningRate)

	-- long story... read on torch website
	if batchSize<10 then
		criterion.sizeAverage = false
	else
		criterion.sizeAverage = true
	end
   
	-- Number of samples
	local nsamples = dataset:size()

	-- make sure our network is in the training mode
	model:training()

	-- move to CUDA
	model = model:cuda()
	criterion = criterion:cuda() 
 
	-- just in case...
	collectgarbage()
 
	-- get model parameters
	parameters,gradParameters = model:getParameters()
	
	-- shuffle before creating mini-batches
	-- Also think whether shuffling is to be removed in case we use LSTMs
	local shuffle = torch.randperm(nsamples, 'torch.LongTensor')

	-- for current iteration, perform batch by batch training procedure
	for t = 1, nsamples, batchSize do
	   
		-- display progress
	        xlua.progress(t, nsamples)

		-- create mini batch
		local inputs  = torch.CudaTensor(batchSize, dataset.data[1]:size()[1], dataset.data[1]:size()[2])
	        local targets = torch.CudaTensor(batchSize)
	        local k = 0
		for i = t,math.min(t+batchSize-1, nsamples) do
			k = k+1
			inputs[k] = dataset.data[shuffle[i]]
			targets[k] = dataset.label[shuffle[i]]
	        end
		
		-- the last mini-batch might not be full..
		inputs = inputs[{{1, k}, {}, {}}]
		targets = targets[{{1, k}}]

		-- closure function for optimization
		local feval = function(x)
			-- get new parameters
			if x~=parameters then
				parameters:copy(x)
			end
			-- reset gradients
			gradParameters:zero()
			-- backpropagate
			local f = criterion:forward(model:forward(inputs), targets)
			model:backward(inputs, criterion:backward(model.output, targets))
			-- return
			return f, gradParameters
		end
	        
		-- do optimization
		optim.adam(feval, parameters, optimConfig)
      
	 end -- end of "for each batch" loop

end

function MBGD:getModel()
	return self.model
end

function MBGD:setLearningRate(learningRate)
	self.learningRate = learningRate
end
