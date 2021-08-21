#!/bin/bash
CUDA_VISIBLE_DEVICES=$2 python train.py  --env TDWdominoes  --model_name GNS --log_per_iter 1000 --training_fpt 3 --ckp_per_iter 5000 --floor_cheat 1  --dataf "$1," --outf "$1_GNS"
