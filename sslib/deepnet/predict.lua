-------------------------------------------------------------------------------
-- Make predictions on unseen data given in .csv format.
-- @arg1 path to trained model to be used for predicting
-- @arg2 path to .csv file
-- @arg3 path to file to save predictions
-------------------------------------------------------------------------------
require 'cunn'
require 'paths'
require 'csvigo'
local inout = require 'sslib.deepnet.lib.inout'
local eval  = require 'sslib.deepnet.lib.eval'
local util  = require 'sslib.deepnet.lib.util'

model_path = arg[1]
dataset_path = arg[2]
output_path = arg[3]

print("## Load trained model")
network = torch.load(model_path)

print("## Load data set")
dataset = inout.load_dataset(dataset_path,3,0)

print("## Generate predictions")
predictions = eval.predict(dataset,network)

print("## Write predictions into a file")
torch.save(output_path,util.toCSV(predictions),'ascii')
