rm -f /home/sleep/data/results/artifacts.txt
CUDA_VISIBLE_DEVICES=0 th NNpredict.lua
# echo "It should have the filename"
# echo $1
# CUDA_VISIBLE_DEVICES=0 th NNpredict.lua $1
# echo "Done with Djordje's predictions"
