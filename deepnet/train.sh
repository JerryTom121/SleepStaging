CUDA_VISIBLE_DEVICES=$2 th NNtrainer.lua $1 | tee logs/training_output_$1
