------------------------------------------------------------------------------
-- This script loads specified and previously trained model and then evaluates
-- its performance on the selected data set
--
-- @param => the configuration of the model/optimization procedure/data set 
---------------------------------------------------------------------------
require 'cunn'
require 'paths'
local inout = require 'lib.inout'
local debug = require 'lib.debug'
local eval  = require 'lib.eval'

------------------------------
-- Parse command line argument
------------------------------
local configFile = arg[1]
assert(configFile=="spectral" or configFile=="temporal" or configFile=="hybrid","ERROR: No existing model is selected!")

------------------------------
-- Fetch experiment parameters
------------------------------
paths.dofile('configs/' .. configFile .. '.lua')
experiment   = getExperiment()
network      = getModel()        -- just for the sake of debugging
optimization = getOptimization() -- just for the sake of debugging 

---------------------
-- Load trained model
---------------------
print('## Loading trained model..')
network = torch.load('models/'..configFile)

-----------------------
-- Print the parameters
-----------------------
debug.outputParameters(network,optimization,experiment)

-----------------------
-- Load validation data
-----------------------
print("## Load validation data...")
if configFile=="hybrid" then
	-- load
	testSet = inout.load_dataset('../../CSV/test_exp'..experiment.number..'.csv',1,3)
	-- split into temporal and fft matrix
	CONV_test = torch.reshape(testSet.data[{{},{1,experiment.numChan*experiment.epochSize}}],
				   testSet.data:size(1),experiment.numChan,experiment.epochSize):transpose(2,3)
	FFT_test  = testSet.data[{{},{experiment.numChan*experiment.epochSize+1,experiment.numChan*experiment.epochSize+experiment.numFFTfeat}}]
	-- print testing set dimensions 
	print("Conv module input dimensions: "..CONV_test:size(1).." x "..CONV_test:size(2).." x "..CONV_test:size(3))
	print("FFT module input dimensions:  "..FFT_test:size(1).." x "..FFT_test:size(2))
	-- Move to CUDA
        testSet.data = {}
        testSet.data.convData = CONV_test:cuda()
        testSet.data.fftData  = FFT_test:cuda()
        testSet.label         = testSet.label:cuda()
        -- Enable access
        setmetatable(testSet.data,
        	     {__index = function(t, i)
                		     return {t.fftData[i],t.convData[i]}
                                end}
        );
else
        testSet = inout.load_dataset('../../CSV/test_exp'..experiment.number..'.csv',experiment.numChan,3)
        print("## Input dimensions: "..testSet.data:size(1).." x "..testSet.data:size(2))
end

---------------------------------
-- Evaluate accuracy of the model
---------------------------------
eval.test(testSet,network)
