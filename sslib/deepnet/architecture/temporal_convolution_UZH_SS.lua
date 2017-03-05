-------------------------------------------------------------------------------
-- Setting up the training procedure for the NN model
-- which uses raw temporal signal as an input with 3 
-- channels 2 EEG and 1 EMG
-------------------------------------------------------------------------------

-- CONSTANTS
EPOCH_SIZE = 512
NUM_CHAN   = 3

-------------------------------------------------------------------------------
-- Optimization procedure parameters
-------------------------------------------------------------------------------
function getOptimization()
        -- change parameters here
        local optimization = {}
        optimization.learningRate      = 0.0001
        optimization.learningRateDecay = 0
        optimization.momentum          = 0
        optimization.batchSize         = 1
        optimization.iterations        = 3
        optimization.classes           = {1,2,3}
        return optimization
end

-------------------------------------------------------------------------------
-- Neural network architecture
-------------------------------------------------------------------------------
function getModel()

    -- Layer 1 (temporal convolution) parameters
    CONV_L1_featureMaps = 30
    CONV_L1_kernel      = 65
    CONV_L1_stride      = 1

    -- MP1
    MP_L1_region = 4
    MP_L1_stride = 4

    -- Layer 2 (temporal convolution) parameters
    CONV_L2_featureMaps = 30
    CONV_L2_kernel      = 33
    CONV_L2_stride      = 1

    -- MP2
    MP_L2_region = 2
    MP_L2_stride = 2

    -- FC layers
    denseNetwork = 500

    -- feature extractor
    convnet = nn.Sequential()
    signal = EPOCH_SIZE; print(signal)

    -- Add Layer 1:
    convnet:add(nn.TemporalConvolution(NUM_CHAN,CONV_L1_featureMaps,CONV_L1_kernel,CONV_L1_stride))
    convnet:add(nn.ReLU())
    signal  = (signal-CONV_L1_kernel)/CONV_L1_stride + 1; print(signal)

    -- Add Max pooling layer
    convnet:add(nn.TemporalMaxPooling(MP_L1_region,MP_L1_stride))
    signal =  (signal-MP_L1_region)/MP_L1_stride+1; print(signal)

    -- Add Layer 2
    convnet:add(nn.TemporalConvolution(CONV_L1_featureMaps,CONV_L2_featureMaps,CONV_L2_kernel,CONV_L2_stride))
    convnet:add(nn.ReLU())
    signal  = (signal-CONV_L2_kernel)/CONV_L2_stride + 1; print(signal)

    -- Add Max pooling layer
    convnet:add(nn.TemporalMaxPooling(MP_L2_region,MP_L2_stride))
    signal =  (signal-MP_L2_region)/MP_L2_stride+1; print(signal)

    -- normalize view
    convnet:add(nn.View(CONV_L2_featureMaps*signal))
    -- full network architecture
    net = nn.Sequential()
            :add(convnet)
	    :add(nn.Dropout(0.5))
	    :add(nn.Linear(CONV_L2_featureMaps*signal,denseNetwork))
	    :add(nn.ReLU())                                
	    :add(nn.Linear(denseNetwork,#getOptimization().classes))
	    :add(nn.LogSoftMax())
    return net
end
