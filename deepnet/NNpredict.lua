---------------------------------------------------------------------------
-- The main script for artifact prediction
---------------------------------------------------------------------------
require 'cunn'
require 'paths'
require 'csvigo'
require 'thcsv'
local inout = require 'lib.inout'
local debug = require 'lib.debug'
local eval  = require 'lib.eval'



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

----------------------
-- Chose trained model
----------------------
network = torch.load('models/temporal_convolution')

----------------
-- Load CSV file
----------------
data = inout.load_dataset('../../CSV/temporal_data.csv',3,0)

-----------------------
-- Generate predictions
-----------------------
predictions = eval.predict(data,network)

-------------------
-- Save predictions
-------------------
torch.save('../../CSV/results.txt',toCSV(predictions),'ascii')
