require 'torch'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
local util = require 'sslib.deepnet.lib.util'

local MBGD = torch.class('nn.MBGD')

function MBGD:__init(optimization, model, dataset)
	-- optimization pars
	self.learningRate = optimization.learningRate
	self.batchSize = optimization.batchSize or 1
	self.weightDecay = optimization.weightDecay or 0
	self.momentum = optimization.momentum or 0
	self.classes = optimization.classes
	-- model
	self.model = model
	-- data
	self.dataset = dataset
	-- criterion (class-balanced)
	class_weights = util.get_class_weights(self.dataset, self.classes)
	self.criterion = nn.ClassNLLCriterion(class_weights)
end



function MBGD:train()

	-- Initalization
	local criterion = self.criterion
	local batchSize = self.batchSize
	local model = self.model
	local classes = self.classes
	local dataset = self.dataset
	local optimConfig = {
		learningRate = self.learningRate,
		weightDecay = self.weightDecay,
		momentum = self.momentum
	}

	print("## Learning rate is "..optimConfig.learningRate)

	-- for now we allow batch size = 1 only: TO BE FIXED
	assert(batchSize==1)

	-- long story... read on torch website
	criterion.sizeAverage = false
   
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
	
	-- intialize timer
	local time = sys.clock()
      
	-- shuffle before creating mini-batches
	-- Also remove shuffling when introducing recurrency !!!
	local shuffle = torch.randperm(nsamples, 'torch.LongTensor')

	-- for current iteration, perform batch by batch training procedure
	for t = 1, nsamples, batchSize do
	   
		-- display progress
	        xlua.progress(t, nsamples)

		-- Create mini batch
		local inputs  = {}
	        local targets = {}
	        for i = t,math.min(t+batchSize-1, nsamples) do
	        	local input  = dataset.data[shuffle[i]]
			local target = dataset.label[shuffle[i]]
		        table.insert(inputs, input)
		        table.insert(targets, target)
	        end

		-- closure function for optimization
		local feval = function(x)
			-- get new parameters
			if x~=parameters then
				parameters:copy(x)
			end
			-- reset gradients
			gradParameters:zero()
			-- f is the average of all criterions
			local f = 0
			for i = 1, #inputs do
				-- process current sample of the batch
				f = f + criterion:forward(model:forward(inputs[i]), targets[i][1])
				model:backward(inputs[i], criterion:backward(model.output, targets[i][1]))
			end
			-- normalize gradients and f(X)
			gradParameters:div(#inputs)
			f = f/#inputs
			-- return
			return f, gradParameters
		end
	        
		-- Do optimization
	        optim.sgd(feval, parameters, optimConfig)
      
	 end -- end of "for each batch" loop

      -- time taken
      time = sys.clock() - time
      print ("==> time to learn 1 sample = "..(time/nsamples*100)..'ms')

end

function MBGD:getModel()
	return self.model
end

function MBGD:setLearningRate(learningRate)
	self.learningRate = learningRate
end
