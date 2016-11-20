-----------------------------------------------------
-- Setting up the training procedure for the NN model
-- which uses raw temporal signal as an input with 3 
-- channels 2 EEG and 1 EMG
-----------------------------------------------------


-- CONSTANTS
EPOCH_SIZE = 128
NUM_CHAN   = 3


------------------------------
-- Neural network architecture
------------------------------
function getModel()
	 -- Layer 1 (temporal convolution) parameters
	 CONV_L1_featureMaps = 15
	 CONV_L1_kernel      = 9
	 CONV_L1_stride      = 1
	 -- MaxPool 1
	 MP_L1_region        = 2
	 MP_L1_stride        = 2
	 -- Layer 2 (temporal convolution) parameters
	 CONV_L2_featureMaps = 15
	 CONV_L2_kernel      = 9
	 CONV_L2_stride      = 1
	 -- MaxPool 2
	 MP_L2_region        = 2
	 MP_L2_stride        = 2
	 -- Layer 3 (temporal convolution) parameters
	 CONV_L3_featureMaps = 15
	 CONV_L3_kernel      = 9
	 CONV_L3_stride      = 1
	 -- FC layer
	 denseNetwork       = 500
	 -- feature extractor
	 convnet = nn.Sequential()
	 signal = EPOCH_SIZE
	 print(signal)
	 -- Add Layer 1:
	 convnet:add(nn.TemporalConvolution(NUM_CHAN,CONV_L1_featureMaps,CONV_L1_kernel,CONV_L1_stride))
	 convnet:add(nn.ReLU())
	 signal  = (signal-CONV_L1_kernel)/CONV_L1_stride + 1; print(signal)
	 -- Add Max pooling layer
	 -------convnet:add(nn.TemporalMaxPooling(MP_L1_region,MP_L1_stride))
	 -------signal =  (signal-MP_L1_region)/MP_L1_stride+1
	 -------print(signal)
	 -- Add Layer 2
	 convnet:add(nn.Dropout(0.5))
	 convnet:add(nn.TemporalConvolution(CONV_L1_featureMaps,CONV_L2_featureMaps,CONV_L2_kernel,CONV_L2_stride))
	 convnet:add(nn.ReLU())
	 signal  = (signal-CONV_L2_kernel)/CONV_L2_stride + 1; print(signal)
	 -- Add Max pooling layer:
	 ------convnet:add(nn.TemporalMaxPooling(MP_L2_region,MP_L2_stride))
	 ------signal =  (signal-MP_L2_region)/MP_L2_stride+1
	 ------print(signal)
	 -- add Layer 3:
	 convnet:add(nn.Dropout(0.5))
	 convnet:add(nn.TemporalConvolution(CONV_L2_featureMaps,CONV_L3_featureMaps,CONV_L3_kernel,CONV_L3_stride))
         convnet:add(nn.ReLU())
         signal  = (signal-CONV_L3_kernel)/CONV_L3_stride + 1; print(signal)
	 -- normalize view
	 convnet:add(nn.View(CONV_L3_featureMaps*signal))
	
	 -- full network architecture
	 net = nn.Sequential()
	         :add(convnet)
	         :add(nn.Dropout(0.5))
	         :add(nn.Linear(CONV_L3_featureMaps*signal,denseNetwork))
	         :add(nn.ReLU())                                 -- :add(nn.Threshold(0, 1e-6)) instead of RELU?????
	         :add(nn.Linear(denseNetwork,2))
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
	optimization.learningRate      = 0.001
	optimization.learningRateDecay = 0
	optimization.batchSize         = 100
	optimization.iterations        = 10

	return optimization
end


------------------------------------------
-- Experiment parameters (define data set)
------------------------------------------
function getExperiment()
	-- change parameters here
	local experiment = {}
	experiment.number   = 1
	experiment.numChan  = 3
	experiment.augType = '_rot_mir'
	experiment.name    = 'temporal'
	experiment.normalize = false
	
	return experiment
end
