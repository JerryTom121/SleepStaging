-------------------------------------------------------------------------------
-- Neural network architecture
-------------------------------------------------------------------------------

-- Layer 1 (temporal convolution) parameters
CONV_L1_featureMaps = 25
CONV_L1_kernel      = 65
CONV_L1_stride      = 1

-- MP1
MP_L1_region = 2
MP_L1_stride = 2

-- Layer 2 (temporal convolution) parameters
CONV_L2_featureMaps = 20
CONV_L2_kernel      = 25
CONV_L2_stride      = 1

-- MP2
MP_L2_region = 2
MP_L2_stride = 2

-- FC layers
denseNetwork = 2000

-- feature extractor
convnet = nn.Sequential()
signal = params.inputSize; print(signal)

-- Add Layer 1:
convnet:add(nn.TemporalConvolution(params.numChannels,CONV_L1_featureMaps,CONV_L1_kernel,CONV_L1_stride))
       --:add(nn.BatchNormalization(CONV_L1_featureMaps))
       :add(nn.ReLU())
signal  = (signal-CONV_L1_kernel)/CONV_L1_stride + 1; print(signal)

-- Add Max pooling layer
convnet:add(nn.TemporalMaxPooling(MP_L1_region,MP_L1_stride))
signal =  (signal-MP_L1_region)/MP_L1_stride+1; print(signal)

-- Add Layer 2
convnet:add(nn.TemporalConvolution(CONV_L1_featureMaps,CONV_L2_featureMaps,CONV_L2_kernel,CONV_L2_stride))
       --:add(nn.BatchNormalization(CONV_L2_featureMaps))
       :add(nn.ReLU())
signal  = (signal-CONV_L2_kernel)/CONV_L2_stride + 1; print(signal)

-- Add Max pooling layer
convnet:add(nn.TemporalMaxPooling(MP_L2_region,MP_L2_stride))
signal =  (signal-MP_L2_region)/MP_L2_stride+1; print(signal)

-- normalize view
convnet:add(nn.View(CONV_L2_featureMaps*signal))

-- full network architecture
model = nn.Sequential()
          :add(convnet)
	  :add(nn.Dropout(params.dropout))
	  :add(nn.Linear(CONV_L2_featureMaps*signal,denseNetwork))
	  --:add(nn.BatchNormalization(denseNetwork))
	  :add(nn.ReLU())
	  :add(nn.Dropout(params.dropout))
          :add(nn.Linear(denseNetwork,denseNetwork))
	  --:add(nn.BatchNormalization(denseNetwork))
          :add(nn.ReLU())
	  :add(nn.Linear(denseNetwork,params.nclasses))
	  :add(nn.LogSoftMax())
