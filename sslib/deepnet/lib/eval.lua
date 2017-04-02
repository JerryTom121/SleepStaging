-------------------------------------------------------------------------------
-- Script Var
-------------------------------------------------------------------------------
local M = {};

require 'math'
require 'optim'

-------------------------------------------------------------------------------
-- Make predictions on a given test data set
-- @param testset test data set
-------------------------------------------------------------------------------
function M.predict(testset, network)

    -- make sure the network is in evaluation mode
    network:evaluate()

    -- predictions array
    local predictions = {}

    -- predict label for each sample of the test data
    for i=1, testset:size() do
        local val, ind = torch.max(network:forward(testset.data[i]), 1)
        predictions[i] = ind[1]
    end

    -- return
    return predictions

end

-------------------------------------------------------------------------------
-- Make and evaluate predictions on a given test data set
-- @param testset test data set
-------------------------------------------------------------------------------
function M.predict_and_evaluate(testset, network)

    -- make sure the network is in evaluation mode
    network:evaluate()

    -- initialization
    local ncorrect = 0
    local confusion = optim.ConfusionMatrix({1, 2, 3, 4})

    -- iterate through the dataset
    for i=1, testset:size() do
	local val, ind = torch.max(network:forward(testset.data[i]), 1)
	-- check whether our prediction is good
	if ind[1] == testset.label[i][1] then
		ncorrect = ncorrect + 1
	end
	-- update confusion matrix
	confusion:add(ind[1], testset.label[i][1])
    end
    
    -- print confusion matrix
    print(confusion)

    -- return
    return ncorrect/testset:size()

end
