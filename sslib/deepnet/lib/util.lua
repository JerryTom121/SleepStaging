local M = {};

function M.get_class_weights(dataset)
    
    class_weights = torch.Tensor(2)
    class_weights[1] = dataset:size() / dataset.label:eq(-1):sum()
    class_weights[2] = dataset:size() / dataset.label:eq(1):sum()

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
