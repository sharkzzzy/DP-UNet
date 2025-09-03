echo $0
export CUDA_VISIBLE_DEVICES=$0
CUDA_VISIBLE_DEVICES=0 python train_supervisionv2.py -c config/vaihingen/mambaunet.py | tee -a output_vaihingen.txt 
CUDA_VISIBLE_DEVICES=0 python vaihingen_test.py -c config/vaihingen/mambaunet.py -o fig_results/vaihingen/mamba --val --rgb -t 'd4' | tee -a output_vaihingen_test.txt