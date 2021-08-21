#!/bin/bash
# train GNS
CUDA_VISIBLE_DEVICES=$1 python train.py  --env TDWdominoes --model_name GNS --log_per_iter 1000 --training_fpt 3 --ckp_per_iter 5000 --floor_cheat 1  --dataf "towers, collision, dominoes, linking, rollingSliding, drop, containment, clothSagging" --outf "all_GNS"

# train DPI
#CUDA_VISIBLE_DEVICES=$1 python train.py  --env TDWdominoes  --model_name GNSRigidH --log_per_iter 1000 --training_fpt 3 --ckp_per_iter 5000 --floor_cheat 1 --dataf "towers, collision, dominoes, linking, rollingSliding, drop, containment, clothSagging" --outf "all_DPI"