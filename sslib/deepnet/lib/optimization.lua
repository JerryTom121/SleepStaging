require 'torch'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

local MBGD = torch.class('nn.MBGD')



function MBGD:__init(module, criterion, optimization, name, classes)
   self.module 		  = module
   self.criterion 	  = criterion
   self.learningRate 	  = optimization.learningRate
   self.learningRateDecay = optimization.learningRateDecay
   self.maxIteration 	  = optimization.iterations
   self.batchSize 	  = optimization.batchSize
   self.optbalance        = optimization.balanced
   self.weightDecay 	  = optimization.weightDecay or 0
   self.momentum 	  = 0
   self.name		  = name
   self.classes           = classes
end



function MBGD:train(trainSet)

   -- init
   local iteration = 1
   local model = self.module
   local criterion = self.criterion
   local classes = self.classes

   criterion.sizeAverage = false
   
   -- Temporary solution to speed up learning 
   -- to be used only before recurrency is introduced!!!
   -- we remove all rows which contain label '4'-artifact
   -- and an ambigious scorings - '5'
   print('Before it was: '..trainSet:size())
   selected = trainSet.label:lt(4):nonzero()[{{},1}]
   trainSet.data = trainSet.data:index(1,selected)
   trainSet.label = trainSet.label:index(1,selected)

   -- Number of samples
   local nsamples = trainSet:size()
   print('Number of samples is '..nsamples)

   -- confusion
   confusion = optim.ConfusionMatrix(classes)

   -- make sure our network is in the training mode
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
   if self.batchSize>1 then
	   print("## Mini Batch Gradient Descent training started...")
	   print("Batch size = "..self.batchSize)
   else 
	   print("## Stochastic Gradient Descent training started...")
   end

   -- Iterate until specified
   while true do
     
      local time = sys.clock()
      
      -- shuffle before creating mini-batches
      -- Also remove shuffling when introducing recurrency !!!
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
		for i = 1,#inputs do
			-- process current sample of the batch
			f = f + criterion:forward(model:forward(inputs[i]),targets[i])
			model:backward(inputs[i],criterion:backward(model.output,targets[i]))
                	confusion:add(model.output,targets[i])
		end
		-- normalize gradients and f(X)
		gradParameters:div(#inputs)
		f = f/#inputs
		-- return
		return f,gradParameters
	 end
		
       -- do the optimization for the current mini-batch			
       --optim.adagrad(feval, parameters, optimState)
       optim.sgd(feval, parameters, optimState) -- with momentum
       --optim.adam(feval, parameters, optimState) -- with momentum
       
      end -- end of "for each batch" loop

      -- time taken
      time = sys.clock() - time
      print ("\n==> overall time = "..(time*100000)..'s')
      print ("\n==> time to learn 1 sample = "..(time/nsamples*100)..'ms')

      -- confusion matrix
      print(confusion)
      confusion:zero()

      -- save/log current net
      local filename = 'models/'..self.name..'_iter='..iteration
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving model to '..filename)
      torch.save(filename, model)
	
      -- increase the iiteration number
      iteration = iteration + 1

      -- decay learning rate (pay attention: we use self.learningRateDecay)
      optimState.learningRate = optimState.learningRate/(1+iteration*self.learningRateDecay)

      -- check if max iteration has been reached
      if self.maxIteration > 0 and iteration > self.maxIteration then
           print("# MBGD: you have reached the maximum number of iterations")
--         print("# training error = " .. currentError)
           break
      end

   end -- end of while
end
