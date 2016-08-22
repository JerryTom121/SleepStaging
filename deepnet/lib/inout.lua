local M = {};


--------------------------------------------------------
-- Load the data set in a appropriate Tensor format
-- @param filepath path to file where data set is stored
-- @param nchannels number of input feature maps
-- @param nlabels number of labels per sample
--------------------------------------------------------
function M.load_dataset(filepath,nchannels,nlabels)

	local dataset = {}
	
	-- Read .csv file containing the data set
	local datasetCSV  = readCSV(filepath)

	-- Infer the dimensions of loaded data set	
	local nfeatures = datasetCSV:size(2) - 1 - nlabels
	local nsamples  = datasetCSV:size(1)

	-- Create data sets by discarding first column and using the last one as labels
	dataset.data   =  datasetCSV[{{},{2,datasetCSV:size(2)-nlabels}}]
	dataset.label  =  datasetCSV[{{},{datasetCSV:size(2)-nlabels+1,datasetCSV:size(2)}}]

	-- Reshape data sets to fit the convolutional  network architecture
	dataset.data = torch.reshape(dataset.data,nsamples,nchannels,nfeatures/nchannels)
	dataset.data = dataset.data:transpose(2,3)
	
	-- Some other preparation for training of our convolutional neural network
	dataset.data = dataset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
	setmetatable(dataset,
	    {__index = function(t, i)
	                    return {t.data[i], t.label[i]}
	                end}
	);
	function dataset:size()
	    return self.data:size(1)
	end
	setmetatable(dataset,
	    {__index = function(t, i)
	                    return {t.data[i], t.label[i]}
	                end}
	);
	function dataset:size()
	    return self.data:size(1)
	end

	return dataset
end



----------------------------------------
-- Read data from CSV to tensor
-- @param filepath the path to .csv file
----------------------------------------
function readCSV(filepath)
       -- Read given file
        local csvFile = io.open(filepath, 'r')
        -- Count number of rows and columns in file
        local COLS
        local ROWS
        local i = 0
        for line in io.lines(filepath) do
          if i == 0 then
            COLS = #line:split(',')
          end
          i = i + 1
        end
        ROWS = i
        -- Load file into Tensor
        local data = torch.Tensor(ROWS, COLS)
        local i = 0
        for line in csvFile:lines('*l') do
          i = i + 1
          local l = line:split(',')
          for key, val in ipairs(l) do
	    if val=="w" then
		val=10
	    elseif val=="r" then
		val=20
	    elseif val=="n" then
		val=30	
	    elseif val=="a" then
		val=1
	    elseif val=="'" then
		val=1
	    end
            data[i][key] = val
          end
        end
        -- Close file
        csvFile:close()
        -- Return tensor
        return data
end



------------------------------
-- Split given string
-- @param sep separator to use
------------------------------
function string:split(sep)
        local sep, fields = sep, {}
        local pattern = string.format("([^%s]+)", sep)
        self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
        return fields
end


return M
