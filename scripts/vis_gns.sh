#!/bin/bash
# sh vis_gns.sh [TRAIN_SCENARIO_NAME] [EPOCH] [ITER] [Test SCENARIO_NAME] [GPU_ID]

ScenarioArray=("Dominoes" "Collide" "Support" "Link" "Contain" "Roll" "Drop" "Drape")
ScenarioArray=($4)
for val in ${ScenarioArray[*]}; do
	echo $val
    CUDA_VISIBLE_DEVICES=$5  python eval_vis.py --env TDWdominoes --epoch $2 --iter $3 --model_name GNS --training_fpt 3 --mode "test" --floor_cheat 1 --test_training_data_processing 1 --modelf "$1_GNS"  --dataf "$val"
done
