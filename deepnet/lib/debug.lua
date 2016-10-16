local M = {};


------------------------------------------------------------------
-- Print out different parameters as a part of debugging procedure
------------------------------------------------------------------
function M.outputParameters(network,optimization,experiment)

	print("## The network architecture:")
	print("## -------------------------")
	print(network:__tostring());
	--print("## The network feature extractor:")
	--print(network:get(1):__tostring());
	print("## Optimization parameters:")
	print("## ------------------------")
	print("## learning rate        =  "..optimization.learningRate)
	print("## learning rate decay  =  "..optimization.learningRateDecay)
	print("## number of iterations =  "..optimization.iterations)
	print("## batch size           =  "..optimization.batchSize)
	print("## load pre-trained model? "..tostring(optimization.pretrained))
	print("## Experiment parameters:")
	print("## ----------------------")
	print("## Experiment number   = "..experiment.number)
	print("## Augmentation method = "..experiment.augType)
	print("## number of channels  = "..experiment.numChan)
	print("## ###############################")

end

return M
