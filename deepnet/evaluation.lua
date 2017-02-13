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
model, iter = configFile:match("([^_]+)_([^_]+)")
if not model then
	model = configFile
	model_iter = configFile
else
	model_iter = model.."_"..iter
end

------------------------------
-- Fetch experiment parameters
------------------------------
paths.dofile('configs/' .. model .. '.lua')
experiment   = getExperiment()
network      = getModel()        -- just for the sake of debugging
optimization = getOptimization() -- just for the sake of debugging 

---------------------
-- Load trained model
---------------------
print('## Loading trained model..')
network = torch.load('models/'..model_iter)

-----------------------
-- Print the parameters
-----------------------
debug.outputParameters(network,optimization,experiment)

-----------------------
-- Load validation data
-----------------------
print("## Load validation data...")
if experiment.number == 0 then
	testSet = {}
        testSet.data = {}
        temp = inout.load_dataset('../../CSV/test_exp12.csv',3,3)
        fft  = inout.load_dataset('../../CSV/test_exp14.csv',1,3)
        testSet.data.temp = temp.data:cuda()
        testSet.data.fft  = fft.data:cuda()
        testSet.label     = temp.label:cuda()
        setmetatable(testSet.data,
                     {__index = function(t, i)
                                     return {t.temp[i],t.fft[i]}
                                end}
        );
        function testSet:size()
                return testSet.label:size(1)
        end
else
        testSet = inout.load_dataset('../../CSV/test_exp'..experiment.number..'.csv',experiment.numChan,3)
        print("## Input dimensions: "..testSet.data:size(1).." x "..testSet.data:size(2))
end

---------------------------------
-- Evaluate accuracy of the model
---------------------------------
eval.test(testSet,network)
