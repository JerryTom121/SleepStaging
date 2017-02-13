-----------------------------------------------------
-- Setting up the training procedure for the NN model
-- which uses raw temporal signal as an input with 3 
-- channels 2 EEG and 1 EMG
-----------------------------------------------------


-- CONSTANTS
EPOCH_SIZE = 256
NUM_CHAN   = 3


------------------------------
-- Neural network architecture
------------------------------
function getModel()
	 -- Layer 1 (temporal convolution) parameters
	 CONV_L1_featureMaps = 10
	 CONV_L1_kernel      = 40
	 CONV_L1_stride      = 1
	 -- Layer 2 (temporal convolution) parameters
	 CONV_L2_featureMaps = 15
	 CONV_L2_kernel      = 20
	 CONV_L2_stride      = 1
	 -- FC layers
	 denseNetwork1       = 400
	 denseNetwork2	     = 400
	 -- feature extractor
	 convnet = nn.Sequential()
	 signal = EPOCH_SIZE; print(signal)
	 -- Add Layer 1:
	 convnet:add(nn.TemporalConvolution(NUM_CHAN,CONV_L1_featureMaps,CONV_L1_kernel,CONV_L1_stride))
	 convnet:add(nn.ReLU())
	 signal  = (signal-CONV_L1_kernel)/CONV_L1_stride + 1; print(signal)
	 -- Add Layer 2
	 convnet:add(nn.TemporalConvolution(CONV_L1_featureMaps,CONV_L2_featureMaps,CONV_L2_kernel,CONV_L2_stride))
	 convnet:add(nn.ReLU())
	 signal  = (signal-CONV_L2_kernel)/CONV_L2_stride + 1; print(signal)
	 -- normalize view
	 convnet:add(nn.View(CONV_L2_featureMaps*signal))
	 -- full network architecture
	 net = nn.Sequential()
	         :add(convnet)
	         :add(nn.Dropout(0.75))
	         :add(nn.Linear(CONV_L2_featureMaps*signal,denseNetwork1))
	         :add(nn.ReLU())                                 -- :add(nn.Threshold(0, 1e-6)) instead of RELU?????
		 :add(nn.Dropout(0.75))
	         :add(nn.Linear(denseNetwork1,denseNetwork2))
	         :add(nn.ReLU())                                 -- :add(nn.Threshold(0, 1e-6)) instead of RELU?????
	         :add(nn.Linear(denseNetwork2,2))
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
	optimization.learningRate      = 0.00015
	optimization.learningRateDecay = 0.005
	optimization.momentum	       = 0.6
	optimization.batchSize         = 1
	optimization.iterations        = 20

	return optimization
end


------------------------------------------
-- Experiment parameters (define data set)
------------------------------------------
function getExperiment()
	-- change parameters here
	local experiment = {}
	experiment.number    = 12
	experiment.numChan   = 3
	experiment.augType   = '_rot'
	experiment.name      = 'temporalConvolution'
	experiment.normalize = false
	
	return experiment
end