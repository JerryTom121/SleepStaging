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
cmd:option('-predictionsPath','', 'Where to generate predictions')
cmd:text()
params = cmd:parse(arg)

-- Load previously trained model
model = torch.load(params.trainedModelPath)
model:clearState()
torch.save(params.trainedModelPath, model)
-- Load data to be evaluated
data = inout.load_dataset(params.dataPath, 3, 0)
-- Make predictions
predictions = eval.predict(data, model)
-- Save predictions
torch.save(params.predictionsPath, util.toCSV(predictions), 'ascii')
