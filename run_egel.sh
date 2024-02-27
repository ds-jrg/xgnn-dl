#!/bin/sh

end_length=10
number_of_ces=300
number_of_graphs=10
lambdaone=0.5
lambdaone_fidelity=-0.1
random_seed=0

python main.py "$end_length" "$number_of_ces" "$number_of_graphs" "$iterlambdaoneations" "$lambdaone_fidelity" "$random_seed"
