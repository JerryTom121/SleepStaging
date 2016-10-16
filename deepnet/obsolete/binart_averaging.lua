require 'nn'
require 'paths'
require 'cunn'
require 'gnuplot'
require 'lib.SGD'
require 'lib.MBGD'
local inout = require 'lib.inout'
local eval  = require 'lib.eval'

-------------------------------------------------------
-- Experiment variables
-------------------------------------------------------
numExp 		= '9'    -- Number of experiment
augType		= '_rot' -- Artifact augmentation type

--------------------------------------------------------
-- Constants: not to be changed in general
--------------------------------------------------------
numChan	 	= 3      -- 3 channels: EEG1;EEG2;EMG	
numLabels 	= 3	 -- number of testing data labels
epochSize 	= 128    -- length of epoch signal
numFFTfeat      = 13     -- number of fourier features

MP_L2_region        = 3
MP_L2_stride        = 2

------------------------------------------------------
-- Load two networks: fourier and temporal convolution
------------------------------------------------------
fourier_net = torch.load('models/fourier_binart')
tempcon_net = torch.load('models/temp_binart')

----------------------------------------------------------------
-- Load testing data sets from .CSV files of selected experiment
----------------------------------------------------------------
print("## Loading validation data...:")
testSetFFT  = inout.load_dataset('../../CSV/test_exp5.csv',1,numLabels)
testSetTMP  = inout.load_dataset('../../CSV/test_exp1.csv',3,numLabels)

testSetFFT.data  = testSetFFT.data:cuda()
testSetFFT.label = testSetFFT.label:cuda()
testSetTMP.data  = testSetTMP.data:cuda()
testSetTMP.label = testSetTMP.label:cuda()
print("Validation data loaded")

----------------------------------------------------------------
-- Test the accuracy of the network on the testing data set
----------------------------------------------------------------
print "## Validating artefakt detection..."
eval.test2(testSetTMP,tempcon_net,testSetFFT,fourier_net)
