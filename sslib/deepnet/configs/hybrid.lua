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
	 
-- PARAMETERS
	 CONV_L1_featureMaps = 30
	 CONV_L1_kernel      = 39
	 CONV_L1_stride      = 1
	 --
	 CONV_L2_featureMaps = 10
         CONV_L2_kernel      = 6
         CONV_L2_stride      = 2
         --
         denseNetwork       = 400
         

	 -- TEMPORAL CONVOLUTION
	 -----------------------
	 convnet = nn.Sequential()
	 signal = EPOCH_SIZE; print(signal)
	 -- Add Layer 1:
	 convnet:add(nn.TemporalConvolution(NUM_CHAN,CONV_L1_featureMaps,CONV_L1_kernel,CONV_L1_stride)):add(nn.ReLU())
	 signal  = (signal-CONV_L1_kernel)/CONV_L1_stride + 1; print(signal)
	 -- Add Layer 2:
	 convnet:add(nn.TemporalConvolution(CONV_L1_featureMaps,CONV_L2_featureMaps,CONV_L2_kernel,CONV_L2_stride)):add(nn.ReLU())
         signal  = (signal-CONV_L2_kernel)/CONV_L2_stride + 1; print(signal)
	 -- normalize viev
	 convnet:add(nn.View(-1))
	 -- add linear layer
	 convnet:add(nn.Linear(CONV_L2_featureMaps*signal,denseNetwork)):add(nn.ReLU()):add(nn.Dropout(0.5))

	 -- FOURIER ENERGY
	 ----------------------
	 energynet = nn.Sequential()
	 signal = EPOCH_SIZE; print(signal)
	 -- add linear layer
	 energynet:add(nn.Linear(13,denseNetwork)):add(nn.ReLU()):add(nn.Dropout(0.5))
 
 	 -- MIXTURE NETWORK
	 ------------------
	 mix = nn.ParallelTable()
        	 :add(convnet)
        	 :add(energynet)

	 -- FULL NETWORK
	 ---------------
	 net = nn.Sequential()
	         :add(mix)
		 :add(nn.JoinTable(1))
	         :add(nn.Linear(2*denseNetwork,denseNetwork))
	         :add(nn.ReLU())                                 -- :add(nn.Threshold(0, 1e-6)) instead of RELU?????
		 :add(nn.Dropout(0.5))
		 :add(nn.Linear(denseNetwork,denseNetwork))
	         :add(nn.ReLU())                                 -- :add(nn.Threshold(0, 1e-6)) instead of RELU?????
		 :add(nn.Dropout(0.5))
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
	optimization.pretrained        = true
	optimization.learningRate      = 0.00008 --0.0001
	optimization.learningRateDecay = 0.01
	optimization.momentum	       = 0.5
	optimization.batchSize         = 1
	optimization.iterations        = 20
	optimization.weightDecay       = 0.0005

	return optimization
end


------------------------------------------
-- Experiment parameters (define data set)
------------------------------------------
function getExperiment()
	-- change parameters here
	local experiment = {}
	experiment.number    = 0
	experiment.numChan   = NUM_CHAN
	experiment.augType   = ''
	experiment.name      = 'hybrid'
	experiment.normalize = false
	
	return experiment
end
