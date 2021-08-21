#!/bin/bash
# sh eval_gns_ransac.sh [TRAIN_SCENARIO_NAME] [EPOCH] [ITER] [Test SCENARIO_NAME] [GPU_ID]

ScenarioArray=("Dominoes" "Collide" "Support" "Link" "Contain" "Roll" "Drop" "Drape")
ScenarioArray=($4)
for val in ${ScenarioArray[*]}; do
	echo $val
    CUDA_VISIBLE_DEVICES=$5  python eval.py --subsample 1000 --env TDWdominoes --ransac_on_pred 1 --vis_only_fail 0 --epoch $2 --iter $3 --model_name GNS --training_fpt 3 --mode "test" --floor_cheat 1 --test_training_data_processing 1 --modelf "$1_GNS"  --dataf "$val"
done

