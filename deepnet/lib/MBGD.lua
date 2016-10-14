require 'torch'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

local MBGD = torch.class('nn.MBGD')



function MBGD:__init(module, criterion, optimization)
   self.learningRate 	  = optimization.learningRate
   self.learningRateDecay = optimization.learningRateDecay
   self.maxIteration 	  = optimization.iterations
   self.batchSize 	  = optimization.batchSize
   self.module 		  = module
   self.criterion 	  = criterion
   self.weightDecay 	  = 0
   self.momentum 	  = 0
end



function MBGD:train(trainSet)

   -- init
   local iteration = 1
   local model = self.module
   local criterion = self.criterion

   -- Number of samples
   local nsamples = trainSet:size()
   print('Number of samples is '..nsamples)

   -- confusion
   confusion = optim.ConfusionMatrix({1,2})

   -- make sure our network is in a training mode
   model:training()

   -- move to CUDA
   model = model:cuda()
   criterion = criterion:cuda() 
 
   -- just in case...
   collectgarbage()
 
   -- get model parameters
   parameters,gradParameters = model:getParameters()

   -- optimization parameters
   optimState = {
      learningRate 	= self.learningRate,
      weightDecay 	= self.weightDecay,
      momentum 		= self.momentum,
      learningRateDecay = 0 			-- pay attention to this, it's kind of weird (e.g. check optim.sgd.lua)
   } 

   -- Output training
   print("## Mini Batch Gradient Descent training started")
   print("Batch size = "..self.batchSize)	

   -- Iterate until specified
   while true do
     
      local time = sys.clock()
      
      -- shuffle before creating mini-batches
      local shuffle = torch.randperm(nsamples, 'torch.LongTensor')

      print("-------------")
      print("Iteration #"..iteration)
      print("Learning rate is: "..optimState.learningRate)

      -- for current iteration, perform batch by batch training procedure
      for t = 1,nsamples,self.batchSize do

	 -- display progress
         xlua.progress(t, nsamples)

	 -- Create mini batch
	 local inputs  = {}
         local targets = {}
         for i = t,math.min(t+self.batchSize-1,nsamples) do
         	local input  = trainSet.data[shuffle[i]]
		local target = trainSet.label[shuffle[i]]
	        table.insert(inputs,input)
	        table.insert(targets,target)
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
				-- evaluate function for complete mini batch
				for i = 1,#inputs do
					-- get label
					if (targets[i][1]<0)   then label = 1 else label = 2 end
					-- process current sample of the batch
					f = f + criterion:forward(model:forward(inputs[i]),label)
					model:backward(inputs[i],criterion:backward(model.output,label))
					-- update confusion matrix
					if (model.output[2]<model.output[1]) then pred = 1  else pred = 2  end
					confusion:add(pred,label)
				end
				-- normalize gradients and f(X)
				gradParameters:div(#inputs)
				f = f/#inputs
				-- return
				return f,gradParameters
	   		 end
		
       -- do the optimization for the current mini-batch			
       --optim.sgd(feval, parameters, optimState) -- with momentum
       optim.adagrad(feval, parameters, optimState)
       
      end -- end of "for each batch" loop

      -- time taken
      time = sys.clock() - time
      time = time / nsamples
      print ("\n==> overall time = "..(time*100000)..'s')
      print ("\n==> time to learn 1 sample = "..(time*100)..'ms')

      -- confusion matrix
      print(confusion)
      confusion:zero()

      -- increase the iiteration number
      iteration = iteration + 1

      -- decay learning rate (pay attention: we use self.learningRateDecay)
      optimState.learningRate = optimState.learningRate/(1+iteration*self.learningRateDecay)

      -- save/log current net
      local filename = '../models/model.net'
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving model to '..filename)
      torch.save(filename, model)

      -- check if max iteration has been reached
      if self.maxIteration > 0 and iteration > self.maxIteration then
           print("# MBGD: you have reached the maximum number of iterations")
--         print("# training error = " .. currentError)
           break
      end

   end -- end of while
end
