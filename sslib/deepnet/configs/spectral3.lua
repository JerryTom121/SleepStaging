-----------------------------------------------------
-- Setting up the training procedure for the NN model
-- which uses energies of different frequency bands.
-- The model has only 13 features
-----------------------------------------------------

-- CONSTANTS
NUM_FFT_FEATURES = 13

------------------------------
-- Neural network architecture
------------------------------
function getModel()
	-- change parameters here
	local layer1 = 200
	local layer2 = 500
	-- feature extractor
	ftextractor = nn.Sequential()
		      	:add(nn.Linear(NUM_FFT_FEATURES,layer1))
			:add(nn.ReLU())
			:add(nn.Dropout(0.5))
			:add(nn.Linear(layer1,layer2))
			:add(nn.ReLU())
			:add(nn.View(-1))
	-- full network
	net = nn.Sequential()
		:add(ftextractor)
		:add(nn.Dropout(0.5))
		:add(nn.Linear(layer2,2))
		:add(nn.LogSoftMax())

	return net
end

------------------------------------
-- Optimization procedure parameters
------------------------------------
function getOptimization()
	-- change parameters here
	local optimization = {}
	optimization.pretrained        = true
	optimization.learningRate      = 0.01
	optimization.learningRateDecay = 0.01
	optimization.batchSize         = 50
	optimization.iterations        = 5

	return optimization
end

------------------------------------------
-- Experiment parameters (define data set)
------------------------------------------
function getExperiment()
	-- change parameters here
	local experiment = {}
	experiment.number   = 5
	experiment.numChan  = 1
	experiment.augType = '_aug'
	experiment.name    = 'spectral'
	experiment.normalize = true
	
	return experiment
end
