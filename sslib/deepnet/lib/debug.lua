local M = {};

------------------------------------------------------------------
-- Print out different parameters as a part of debugging procedure
------------------------------------------------------------------
function M.outputParameters(model, optimization)

	print("## The network architecture:")
	print("## -------------------------")
	print(model:__tostring());
	--print("## The network feature extractor:")
	--print(network:get(1):__tostring());
	print("## Optimization parameters:")
	print("## ------------------------")
	print("## learning rate        =  "..optimization.learningRate)
	print("## learning rate decay  =  "..optimization.learningRateDecay)
	print("## number of iterations =  "..optimization.iterations)
	print("## batch size           =  "..optimization.batchSize)
	print("## weight decay         =  "..(optimization.weightDecay or 0))
	print("## ###############################")

end

return M
