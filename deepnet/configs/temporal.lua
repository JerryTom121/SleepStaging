-----------------------------------------------------
-- Setting up the training procedure for the NN model
-- which uses raw temporal signal as an input with 3 
-- channels 2 EEG and 1 EMG
-----------------------------------------------------

------------------------------
-- Neural network architecture
------------------------------
function getModel()
	 -- Layer 1 (temporal convolution) parameters
	 CONV_L1_featureMaps = 6
	 CONV_L1_kernel      = 20
	 CONV_L1_stride      = 1
	 -- MaxPool 1
	 MP_L1_region        = 2
	 MP_L1_stride        = 2
	 -- Layer 2 (temporal convolution) parameters
	 CONV_L2_featureMaps = 10
	 CONV_L2_kernel      = 10  --10
	 CONV_L2_stride      = 1
	 -- MaxPool 2
	 MP_L2_region        = 2
	 MP_L2_stride        = 2
	
	 -- feature extractor
	 convnet = nn.Sequential()
	 signal = epochSize
	 print(signal)
	 -- Add Layer 1:
	 convnet:add(nn.TemporalConvolution(numChan,CONV_L1_featureMaps,CONV_L1_kernel,CONV_L1_stride))
	 convnet:add(nn.ReLU())
	 signal  = (signal-CONV_L1_kernel)/CONV_L1_stride + 1
	 -- Add Max pooling layer
	 --convnet:add(nn.TemporalMaxPooling(MP_L1_region,MP_L1_stride))
	 --signal =  (signal-MP_L1_region)/MP_L1_stride+1
	 --print(signal)
	 -- Add Layer 2
	 convnet:add(nn.TemporalConvolution(CONV_L1_featureMaps,CONV_L2_featureMaps,CONV_L2_kernel,CONV_L2_stride))
	 convnet:add(nn.ReLU())
	 signal  = (signal-CONV_L2_kernel)/CONV_L2_stride + 1
	 -- Add Max pooling layer:
	 --convnet:add(nn.TemporalMaxPooling(MP_L2_region,MP_L2_stride))
	 --signal =  (signal-MP_L2_region)/MP_L2_stride+1
	 --print(signal)
	 -- Add Layer 3: fully connected layer
	 convnet:add(nn.View(CONV_L2_featureMaps*signal))
	
	 -- full network architecture
	 net = nn.Sequential()
	         :add(convnet)
	         :add(nn.Dropout(0.5))
	         :add(nn.Linear(CONV_L2_featureMaps*signal,50))
	         :add(nn.ReLU())                                 -- :add(nn.Threshold(0, 1e-6)) instead of RELU?????
	         :add(nn.Linear(50,2))
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
	optimization.learningRate      = 0.002
	optimization.learningRateDecay = 0.01
	optimization.batchSize         = 10
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
	
	return experiment
end
