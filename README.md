## Requirements

PyTorch `2.0.1`

PyTorch Geometric `2.3.1`


## Run Experiments

#### F-DIGNN (Example)

python main.py --input chameleon --model Neural_Simplified --lr 0.01 --mu 2.20 --num_hid 128 --weight_decay 0 --dropout 0.5 --add_noise False --base_ratio 0 --extreme_ratio 0


#### L1-DIGNN (Example)

python main.py --input chameleon --model Neural_Simplified_L1 --lr 0.001 --lbd 1.50 --num_hid 64 --weight_decay 1e-3 --dropout 0.5 --K_iter 6 --add_noise False --base_ratio 0 --extreme_ratio 0
