
local M = {};



--------------------------------------------------
-- Evaluate the efficency in discovering artifacts
-- @param dataset testing data set
--------------------------------------------------
function M.test(dataset)
    local hits = 0
    local fp = 0
    local fn = 0

    local fp_doubtful  = 0
    local fn_neighbour = 0

    for i=1,dataset:size() do
        local groundtruth = dataset.label[i][1]
        local prediction = net:forward(dataset.data[i])

        -- artefakt
        if groundtruth==-1 then
            -- detected
            if prediction[1]<0 then
                hits = hits + 1
            -- not detected
            else
                fn = fn + 1
                -- Check if the missed artefakt is in 
                -- the neighbourhood of another artefakt
               --  if dataset.label[i][4]==-2 then
               --         fn_neighbour = fn_neighbour + 1
               --  end
            end
        else
            -- false positive
            if prediction[1]<0 then
                fp = fp + 1
                -- check if this false positive has been classified as
                -- an artifact at least by one of the raters
                -- if dataset.label[i][2]==-1 or dataset.label[i][3]==-1 then
                --         fp_doubtful = fp_doubtful + 1
                -- end
            end
        end
    end
    print('Number of guessed artefakts: '..hits)
    print('Number of false positives:   '..fp)
    print('Number of missed artefakts:  '..fn)
    print('----------------------------------------')
    print('Number of doubtful false positives: '..fp_doubtful)
    print('Number of misclassified artifacts which are in the neighbourhood of another: '..fn_neighbour)
end

return M
