require 'cunn'
require 'paths'
require 'csvigo'
local inout = require 'lib.inout'


nnmodel_path = arg[1]
feature_path = arg[2]

print(nnmodel_path)
print(feature_path)

print("## Load trained model")
network = torch.load(nnmodel_path)

print(network)
