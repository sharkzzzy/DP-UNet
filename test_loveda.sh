echo $0
export CUDA_VISIBLE_DEVICES=$0
CUDA_VISIBLE_DEVICES=0 python loveda_test.py -c config/loveda/mambaunet.py -o fig_results/loveda/mamba --val --rgb -t 'd4' | tee -a output_loveda_test.txt