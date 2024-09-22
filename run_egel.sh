#!/bin/sh

run_DBLP="True"
run_BAShapes="False"
random_seed=1
iterations=5

python main.py "$run_DBLP" "$run_BAShapes" "$random_seed" "$iterations"
