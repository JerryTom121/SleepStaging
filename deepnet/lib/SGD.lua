require 'optim'

local SGD = torch.class('nn.SGD')

function SGD:__init(module, criterion)
   self.learningRate = 0.001
   self.learningRateDecay = 0
   self.maxIteration = 25
   self.shuffleIndices = true
   self.module = module
   self.criterion = criterion
   self.verbose = true
end

function SGD:train(convnet_data,fft_data,labels)
   local iteration = 1
   local currentLearningRate = self.learningRate
   local module = self.module
   local criterion = self.criterion
	
   local nsamples = labels:size(1)
   print('Number of samples is '..nsamples)

   local shuffledIndices = torch.randperm(nsamples, 'torch.LongTensor')
   if not self.shuffleIndices then
      for t = 1,nsamples do
         shuffledIndices[t] = t
      end
   end
   
   -- confusion
   confusion = optim.ConfusionMatrix({1,2})

   print("# SGD: training")

   while true do
      local currentError = 0
      for t = 1,nsamples do
	 local ind = shuffledIndices[t]
	 local input = {convnet_data[{{ind},{},{}}],fft_data[{{ind},{}}]}
	 local target = labels[ind]

         currentError = currentError + criterion:forward(module:forward(input), target)

         module:updateGradInput(input, criterion:updateGradInput(module.output, target))
         module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)

         if self.hookExample then
            self.hookExample(self, example)
         end

	 -- update confusion
         if (module.output[1]<0)  then pred = 1  else pred = 2  end
         if (target[1]<0)         then label = 1 else label = 2 end
         confusion:add(pred,label)

      end

      currentError = currentError / nsamples

      if self.hookIteration then
         self.hookIteration(self, iteration, currentError)
      end

      if self.verbose then
	 print('# learning rate was = '..currentLearningRate)
         print("# current error = " .. currentError)
	 print(confusion)
	 print(confusion.totalValid)
	 confusion:zero()
      end

      iteration = iteration + 1
      currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
      if self.maxIteration > 0 and iteration > self.maxIteration then
         print("# SGD: you have reached the maximum number of iterations")
         print("# training error = " .. currentError)
         break
      end
   end
end
