#!/bin/bash

ScenarioArray=($1)
#("Dominoes" "Collide" "Support" "Link" "Contain" "Roll" "Drop" "Drape")

for scn in ${ScenarioArray[*]}; do
	echo $scn
    python preprocessing_tdw_cheap.py --scenario "$scn" --mode "$2"
done

#python get_label_txt --scenario