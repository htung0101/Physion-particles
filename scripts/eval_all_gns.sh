#!/bin/bash
# sh eval_all_gns.sh [EPOCH] [ITER] [Test SCENARIO_NAME] [GPU_ID]


ScenarioArray=("dominoes" "collision" "towers" "linking" "containment" "rollingSliding" "drop" "clothSagging")
ScenarioArray=($3)
for val in ${ScenarioArray[*]}; do
	echo $val
    CUDA_VISIBLE_DEVICES=$4  python eval.py --env TDWdominoes --vis_only_fail 0 --epoch $1 --iter $2 --model_name GNS --training_fpt 3 --mode "test" --floor_cheat 1 --test_training_data_processing 1 --modelf "all_GNS" --dataf "$val"
done
