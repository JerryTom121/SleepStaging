CUDA_VISIBLE_DEVICES=$2 nohup th binart_$1.lua | tee logs/binart_$1$2.log
