#!/bin/bash


#ScenarioArray=("Dominoes" "Collide" "Support" "Link" "Contain" "Roll" "Drop" "Drape")
ScenarioArray=($1)

# if you want to parallel the script, make sure to set is_not_parallel=0
for scn in ${ScenarioArray[*]}; do
	echo $scn
    python get_label_txt.py --scenario "$scn" --label_filename "labels" --is_not_parallel=0 --mode="$2"
done
