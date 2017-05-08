-------------------------------------------------------------------------------
-- Make predictions on test set
-------------------------------------------------------------------------------

require 'torch'
require 'cunn'
require 'paths'
local inout = require 'sslib.deepnet.lib.inout'
local eval  = require 'sslib.deepnet.lib.eval'
local util  = require 'sslib.deepnet.lib.util'

-- Read and parse command line arguments
cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-trainedModelPath', '', 'Path to previously trained neural net')
cmd:option('-dataPath','', 'Path to .csv file containing input to neural net')
cmd:text()
params = cmd:parse(arg)

-- Load previously trained model
model = torch.load(params.trainedModelPath)
-- Load data to be evaluated
data = inout.load_dataset(params.dataPath, 3, 1)
-- Make and evaluate predictions
eval.predict_and_evaluate(data, model)
