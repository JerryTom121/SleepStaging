---------------------------------------------------------------------------
-- The main script for artifact prediction
---------------------------------------------------------------------------
require 'cunn'
require 'paths'
require 'csvigo'
local inout = require 'lib.inout'
local debug = require 'lib.debug'
local eval  = require 'lib.eval'

--------------------
-- System parameters
--------------------
model_path  = 'models/temporal_convolution'
data_path   = '/home/sleep/data/temporal_data.csv'
output_path = '/home/sleep/data/results/artifacts.txt'



-- Used to escape "'s by toCSV
function escapeCSV (s)
   if string.find(s, '[,"]') then
       s = '"' .. string.gsub(s, '"', '""') .. '"'
   end
   return s
end

function toCSV (tt)
  local s = ""
-- ChM 23.02.2014: changed pairs to ipairs 
-- assumption is that fromCSV and toCSV maintain data as ordered array
   for _,p in ipairs(tt) do
       s = s .. "," .. escapeCSV(p)
   end
   return string.sub(s, 2)      -- remove first comma
end




-- load trained model
print("## Load trained model")
network = torch.load(model_path)
-- load data set
print("## Load data set")
data = inout.load_dataset(data_path,3,0)
-- predict
print("## Generate predictions")
predictions = eval.predict(data,network)
-- save predictions
print("## Write predictions into a file")
torch.save(output_path,toCSV(predictions),'ascii')
