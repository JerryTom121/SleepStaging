require 'nn'
require 'paths'
require 'cunn'
local inout = require 'lib.inout'
local eval  = require 'lib.eval'

--------------------------------------------------------
-- Load two networks: fourier and temporal convolution
------------------------------------------------------
fourier_net = torch.load('models/frtenergy1');  exp1 = 5
--fourier_net = torch.load('models/frtconv1_iter=8');  exp1 = 13
tempcon_net = torch.load('models/tmpconv1_iter=8');  exp2 = 12

----------------------------------------------------------------
-- Load testing data sets from .CSV files of selected experiment
----------------------------------------------------------------
print("## Loading validation data...:")
testSetFFT  = inout.load_dataset('../../CSV/test_exp'..exp1..'.csv',1,3)
--testSetFFT  = inout.load_dataset('../../CSV/test_exp'..exp1..'.csv',3,3)
testSetTMP  = inout.load_dataset('../../CSV/test_exp'..exp2..'.csv',3,3)

testSetFFT.data  = testSetFFT.data:cuda()
testSetFFT.label = testSetFFT.label:cuda()
testSetTMP.data  = testSetTMP.data:cuda()
testSetTMP.label = testSetTMP.label:cuda()

----------------------------------------------------------------
-- Test the accuracy of the network on the testing data set
----------------------------------------------------------------
print "## Validating artefakt detection..."
eval.test2(testSetTMP,tempcon_net,testSetFFT,fourier_net)
