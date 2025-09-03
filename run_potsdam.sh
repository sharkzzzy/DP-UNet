echo $0
export CUDA_VISIBLE_DEVICES=$0
CUDA_VISIBLE_DEVICES=0 python train_supervisionv2.py -c config/potsdam/mambaunet.py | tee -a output_potsdam.txt 
CUDA_VISIBLE_DEVICES=0 python potsdam_test.py -c config/potsdam/mambaunet.py -o fig_results/potsdam/mamba --val --rgb -t 'd4' | tee -a output_potsdam_test.txt