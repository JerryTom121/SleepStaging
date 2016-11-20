rm -f /local/home/djordjem/code/CSV/results.txt
CUDA_VISIBLE_DEVICES=$1 th /local/home/djordjem/code/sleepstaging/deepnet/NNpredict.lua
