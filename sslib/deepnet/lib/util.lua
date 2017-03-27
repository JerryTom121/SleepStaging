local M = {};

function M.get_class_weights(dataset,nclasses)
    
    local class_weights = torch.Tensor(nclasses)
    local sum = 0
	
    for i = 1, nclasses do
        class_weights[i] = dataset:size() / dataset.label:eq(i):sum()
	sum = sum + class_weights[i]
    end
    
    -- Normalize weights so they sum up to 1
    class_weights:div(sum)

    return class_weights
end


-- Used to escape "'s by toCSV
function escapeCSV (s)
   if string.find(s, '[,"]') then
       s = '"' .. string.gsub(s, '"', '""') .. '"'
   end
   return s
end


function M.toCSV (tt)
  local s = ""
-- ChM 23.02.2014: changed pairs to ipairs 
-- assumption is that fromCSV and toCSV maintain data as ordered array
   for _,p in ipairs(tt) do
       s = s .. "," .. escapeCSV(p)
   end
   return string.sub(s, 2)      -- remove first comma
end

return M
