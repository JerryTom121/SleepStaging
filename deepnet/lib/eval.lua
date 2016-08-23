-------------
-- Script Var
-------------
local M = {};


--------------------------------------------------
-- Evaluate the efficency in discovering artifacts
-- @param dataset testing data set
--------------------------------------------------
function M.test(dataset,network,hybrid)

    local hits = 0
    local fp = 0
    local fn = 0

    local fp_doubtful  = 0

    local fn_both_previous = 0
    local fn_a_previous = 0
    local fn_c_previous = 0

    local fn_both_subs = 0
    local fn_a_subs = 0
    local fn_c_subs = 0

    local fn_between = 0

    local fp_w_c = 0
    local fp_r_c = 0
    local fp_n_c = 0
    local fp_artifact_c = 0

    local fp_w_a = 0
    local fp_r_a = 0
    local fp_n_a = 0
    local fp_artifact_a = 0

    local average_hit_confidence  = 0
    local average_miss_confidence = 0

    for i=2,dataset:size()-1 do

	-- Construct input based on the network type
	local input = {}
	if hybrid then
		input = {dataset.convData[{{i},{},{}}],dataset.fftData[{{i},{}}]}
	else
		input = dataset.data[i]
	end

	-- Get the true label and our prediction
        local groundtruth = dataset.label[i][1]
        local prediction = network:forward(input)

	local threshold = 0
	--print "PREDICTION THRESHOLD = "..threshold
	--print "------------------------"

        -- True label is artifact
        if groundtruth==-1 then

            -- Artifact successfully detected
            if prediction[1]<threshold then
                hits = hits + 1
		average_hit_confidence = average_hit_confidence + math.abs(prediction[1])

            -- Artifact not detected
            else
                fn = fn + 1
		average_miss_confidence = average_miss_confidence + math.abs(prediction[1])

		-- Check the previous neighbour of the  missed artifact
		if (dataset.label[i-1][1]==-1) then
			fn_both_previous = fn_both_previous + 1
		end
		if (dataset.label[i-1][2]<10) then
			fn_a_previous = fn_a_previous + 1
		end
		if (dataset.label[i-1][3]<10) then
			fn_c_previous = fn_c_previous + 1
		end
	
		-- Check the subsequent neighbour of the missed artifact
		if (dataset.label[i+1][1]==-1) then
			fn_both_subs = fn_both_subs + 1
		end
		if (dataset.label[i+1][2]<10) then
			fn_a_subs = fn_a_subs + 1
		end
		if (dataset.label[i+1][3]<10) then
			fn_c_subs = fn_c_subs + 1
		end

		-- Check if it's in between
		if (dataset.label[i-1][1]==-1 and dataset.label[i+1][1]==-1) then
			fn_between = fn_between + 1
		end

		-- DEBUG
	--	print("Missed artifact number "..fn..":")
	--	print("intersection ground truth sequence: "..dataset.label[i-1][1].." | "..dataset.label[i][1].." | "..dataset.label[i+1][1])
	--	print("Andrea ground truth sequence:       "..dataset.label[i-1][2].." | "..dataset.label[i][2].." | "..dataset.label[i+1][2])
	--	print("Christine ground truth sequence:    "..dataset.label[i-1][3].." | "..dataset.label[i][3].." | "..dataset.label[i+1][3])
	--	print("predicted sequence:                 "..net:forward({dataset.convData[{{i-1},{},{}}],dataset.fftData[{{i-1},{}}]})[1].."| "..net:forward(input)[1].." | "..net:forward( {dataset.convData[{{i+1},{},{}}],dataset.fftData[{{i+1},{}}]})[1])
            end

	-- True label is not an artifact
        else

            -- False positive
            if prediction[1]<threshold then

		-- increase overall number of false positives
                fp = fp + 1
		
		-- Andrea's rating	
		if dataset.label[i][2] == 10 then
			fp_w_a = fp_w_a + 1
		elseif dataset.label[i][2] == 20 then
			fp_r_a = fp_r_a + 1
		elseif dataset.label[i][2] == 30 then
			fp_n_a = fp_n_a + 1
		else
			fp_artifact_a = fp_artifact_a + 1
		end

		-- Christine's rating
		if dataset.label[i][3] == 10 then
			fp_w_c = fp_w_c + 1
		elseif dataset.label[i][3] == 20 then
			fp_r_c = fp_r_c + 1
		elseif dataset.label[i][3] == 30 then
			fp_n_c = fp_n_c + 1
		else
			fp_artifact_c = fp_artifact_c + 1
		end
            end
        end
    end


    average_hit_confidence  = average_hit_confidence/hits
    average_miss_confidence = average_miss_confidence/fn

    print('Number of guessed artefakts: '..hits)
    print('Number of false positives:   '..fp)
    print('Number of missed artefakts:  '..fn)
    print('----------------------------------------')
    print('The average confidence of guessed artifacts: '..average_hit_confidence)
    print('The average confidence of missed artifacts:  '..average_miss_confidence)
    print('----------------------------------------')
    print('----- FALSE POSITIVES STATISTICS -------')
    print('----------------------------------------')
    print('False positive which Andrea classified as Wake: '..fp_w_a)
    print('False positive which Andrea classified as REM: '..fp_r_a)
    print('False positive which Andrea classified as NREM: '..fp_n_a)
    print('False positive which Andrea classified as ARTIFACT actually: '..fp_artifact_a)
    print('----------------------------------------')
    print('False positive which Christine classified as Wake: '..fp_w_c)
    print('False positive which Christine classified as REM: '..fp_r_c)
    print('False positive which Christine classified as NREM: '..fp_n_c)
    print('False positive which Christine classified as ARTIFACT actually: '..fp_artifact_c)
    print('----------------------------------------')
    print('False positive which was classified as artifact by at least one rater: '..fp_artifact_c+fp_artifact_a)
    print('----------------------------------------')
    print('-----MISSED ARTIFACTS STATISTICS -------')
    print('----------------------------------------')
    print('Previous neighbour was classified as an artifact by both raters '..fn_both_previous)
    print('Previous neighbour was classified as an artifact by Andrea: '..fn_a_previous)
    print('Previous neighbour was classified as an artifact by Christine: '..fn_c_previous)
    print('Previous neighbour was classified as an artifact by one of the raters: '..fn_a_previous+fn_c_previous-fn_both_previous)
    print('----------------------------------------')
    print('Subsequent neighbour was classified as an artifact by both raters '..fn_both_subs)
    print('Subsequent neighbour was classified as an artifact by Andrea: '..fn_a_subs)
    print('Subsequent neighbour was classified as an artifact by Christine: '..fn_c_subs)
    print('Subsequent neighbour was classified as an artifact by one of the raters: '..fn_a_subs+fn_c_subs-fn_both_subs)
    print('----------------------------------------')
    print('According to both raters, artifact is in between two other artifacts '..fn_between)
 


end


--------------------------------------------------
-- Evaluate the efficency in discovering artifacts
-- @param dataset testing data set
--------------------------------------------------
function M.test2(dataset,network1,dataset2,network2)

    local hits = 0
    local fp = 0
    local fn = 0

    local fp_doubtful  = 0

    local fn_both_previous = 0
    local fn_a_previous = 0
    local fn_c_previous = 0

    local fn_both_subs = 0
    local fn_a_subs = 0
    local fn_c_subs = 0

    local fn_between = 0

    local fp_w_c = 0
    local fp_r_c = 0
    local fp_n_c = 0
    local fp_artifact_c = 0

    local fp_w_a = 0
    local fp_r_a = 0
    local fp_n_a = 0
    local fp_artifact_a = 0

    local average_hit_confidence  = 0
    local average_miss_confidence = 0

    for i=2,dataset:size()-1 do

	-- Get the true label and our prediction
        local groundtruth = dataset.label[i][1]
        local prediction1 = network1:forward(dataset.data[i])
	local prediction2 = network2:forward(dataset2.data[i])
	local prediction  = (prediction1[1]+prediction2[1])/2

	-- DEBUG
--	print("prediction="..prediction[1].."; truth="..groundtruth)

        -- True label is artifact
        if groundtruth==-1 then

            -- Artifact successfully detected
            if prediction<0 then
                hits = hits + 1
		average_hit_confidence = average_hit_confidence + math.abs(prediction)

            -- Artifact not detected
            else
                fn = fn + 1
		average_miss_confidence = average_miss_confidence + math.abs(prediction)
		-- Check the previous neighbour of the  missed artifact
		if (dataset.label[i-1][1]==-1) then
			fn_both_previous = fn_both_previous + 1
		end
		if (dataset.label[i-1][2]<10) then
			fn_a_previous = fn_a_previous + 1
		end
		if (dataset.label[i-1][3]<10) then
			fn_c_previous = fn_c_previous + 1
		end
	
		-- Check the subsequent neighbour of the missed artifact
		if (dataset.label[i+1][1]==-1) then
			fn_both_subs = fn_both_subs + 1
		end
		if (dataset.label[i+1][2]<10) then
			fn_a_subs = fn_a_subs + 1
		end
		if (dataset.label[i+1][3]<10) then
			fn_c_subs = fn_c_subs + 1
		end

		-- Check if it's in between
                if (dataset.label[i-1][1]==-1 and dataset.label[i+1][1]==-1) then
                        fn_between = fn_between + 1
                end		

		-- DEBUG
		--print("Missed artifact number "..fn..":")
		--print("intersection ground truth sequence: "..dataset.label[i-1][1].." | "..dataset.label[i][1].." | "..dataset.label[i+1][1])
		--print("Andrea ground truth sequence:       "..dataset.label[i-1][2].." | "..dataset.label[i][2].." | "..dataset.label[i+1][2])
		--print("Christine ground truth sequence:    "..dataset.label[i-1][3].." | "..dataset.label[i][3].." | "..dataset.label[i+1][3])
		--print("predicted sequence:                 "..net:forward(dataset.data[i-1])[1].."| "..net:forward(dataset.data[i])[1].." | "..net:forward(dataset.data[i+1])[1])
            end

	-- True label is not an artifact
        else

            -- False positive
            if prediction<0 then

		-- increase overall number of false positives
                fp = fp + 1
		
		-- Andrea's rating	
		if dataset.label[i][2] == 10 then
			fp_w_a = fp_w_a + 1
		elseif dataset.label[i][2] == 20 then
			fp_r_a = fp_r_a + 1
		elseif dataset.label[i][2] == 30 then
			fp_n_a = fp_n_a + 1
		else
			fp_artifact_a = fp_artifact_a + 1
		end

		-- Christine's rating
		if dataset.label[i][3] == 10 then
			fp_w_c = fp_w_c + 1
		elseif dataset.label[i][3] == 20 then
			fp_r_c = fp_r_c + 1
		elseif dataset.label[i][3] == 30 then
			fp_n_c = fp_n_c + 1
		else
			fp_artifact_c = fp_artifact_c + 1
		end
            end
        end
    end


    average_hit_confidence  = average_hit_confidence/hits
    average_miss_confidence = average_miss_confidence/fn

    print('Number of guessed artefakts: '..hits)
    print('Number of false positives:   '..fp)
    print('Number of missed artefakts:  '..fn)
    print('----------------------------------------')
    print('The average confidence of guessed artifacts: '..average_hit_confidence)
    print('The average confidence of missed artifacts:  '..average_miss_confidence)
    print('----------------------------------------')
    print('----- FALSE POSITIVES STATISTICS -------')
    print('----------------------------------------')
    print('False positive which Andrea classified as Wake: '..fp_w_a)
    print('False positive which Andrea classified as REM: '..fp_r_a)
    print('False positive which Andrea classified as NREM: '..fp_n_a)
    print('False positive which Andrea classified as ARTIFACT actually: '..fp_artifact_a)
    print('----------------------------------------')
    print('False positive which Christine classified as Wake: '..fp_w_c)
    print('False positive which Christine classified as REM: '..fp_r_c)
    print('False positive which Christine classified as NREM: '..fp_n_c)
    print('False positive which Christine classified as ARTIFACT actually: '..fp_artifact_c)
    print('----------------------------------------')
    print('False positive which was classified as artifact by at least one rater: '..fp_artifact_c+fp_artifact_a)
    print('----------------------------------------')
    print('-----MISSED ARTIFACTS STATISTICS -------')
    print('----------------------------------------')
    print('Previous neighbour was classified as an artifact by both raters '..fn_both_previous)
    print('Previous neighbour was classified as an artifact by Andrea: '..fn_a_previous)
    print('Previous neighbour was classified as an artifact by Christine: '..fn_c_previous)
    print('Previous neighbour was classified as an artifact by one of the raters: '..fn_a_previous+fn_c_previous-fn_both_previous)
    print('----------------------------------------')
    print('Subsequent neighbour was classified as an artifact by both raters '..fn_both_subs)
    print('Subsequent neighbour was classified as an artifact by Andrea: '..fn_a_subs)
    print('Subsequent neighbour was classified as an artifact by Christine: '..fn_c_subs)
    print('Subsequent neighbour was classified as an artifact by one of the raters: '..fn_a_subs+fn_c_subs-fn_both_subs)
    print('----------------------------------------')
    print('According to both raters, artifact is in between two other artifacts '..fn_between)

end



return M
