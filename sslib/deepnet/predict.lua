-------------------------------------------------------------------------------
-- Make predictions on test set
--
-- @arg1 path to trained model to be used for predicting
-- @arg2 path to .csv file
-- @arg3 path to file to save predictions
-------------------------------------------------------------------------------

require 'cunn'
require 'paths'
local inout = require 'sslib.deepnet.lib.inout'
local eval  = require 'sslib.deepnet.lib.eval'
local util  = require 'sslib.deepnet.lib.util'

-- Command line argument parsing
modelpath = arg[1]
testsetpath = arg[2]
outputpath = arg[3]

print("## Load trained model")
model = torch.load(modelpath)

print("## Load test set")
testset = inout.load_dataset(testsetpath, 3, 0)

print("## Generate predictions")
predictions = eval.predict(testset, model)

print("## Write predictions into a file")
torch.save(outputpath, util.toCSV(predictions), 'ascii')
