-----------------------------------------------------
-- Setting up the training procedure for the NN model
-- which uses a parally connected spectral and 
-- temporal NN models integrated in a hybrid 
-- architecture
-----------------------------------------------------

-- Constants
EPOCH_SIZE       = 128
NUM_FFT_FEATURES = 13

------------------------------
-- Neural network architecture
------------------------------
function getModel()
	-- set parameters appropriately
	extractors_output = 1000+50
	-- load feature extractors
	fourier_extractor = torch.load('models/spectral_extractor')
	tempcon_extractor = torch.load('models/temporal_extractor')
	-- merge them into the hybrid network
	net = nn.Sequential()
	        :add(nn.ParallelTable()
                	:add(fourier_extractor)
                	:add(tempcon_extractor)
        	)
        	:add(nn.JoinTable(1))
	        :add(nn.Dropout(0.5))
	        :add(nn.Linear(1000+50,300))
	        :add(nn.ReLU())
	        :add(nn.Linear(300,2))
	        :add(nn.LogSoftMax())

	return net
end

------------------------------------
-- Optimization procedure parameters
------------------------------------
function getOptimization()
	-- change parameters here
	local optimization = {}
	optimization.pretrained        = false
	optimization.learningRate      = 0.01
	optimization.learningRateDecay = 0.01
	optimization.batchSize         = 10
	optimization.iterations        = 5

	return optimization
end

------------------------------------------
-- Experiment parameters (define data set)
------------------------------------------
function getExperiment()
	-- change parameters here
	local experiment = {}
	experiment.number   = 9
	experiment.numChan  = 3
	experiment.augType = '_rot'
	experiment.epochSize  = EPOCH_SIZE
	experiment.numFFTfeat = NUM_FFT_FEATURES
	
	return experiment
end
