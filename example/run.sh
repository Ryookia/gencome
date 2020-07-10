#!/bin/sh

python ../scripts/gencome_runner.py --x_file_path "./x_java_files.csv" \
	--y_file_path "./y_java_tests_eloc.csv" \
	--max_tree_depth 2 \
	--tournament_size 5 \
	--population_size 100 \
	--crossover_prob 0.95 \
	--mutate_prob 0.1 \
	--generations 500 \
	--results_dir_path "./results"

