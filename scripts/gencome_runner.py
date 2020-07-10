
description="""Runs the process of finding count-based measures."""

import sys
import os
import numpy as np
import pandas as pd
import argparse
import operator
import pickle

from deap import algorithms, creator, base, gp, tools

import multiprocessing

import gencome.config 
from gencome.gp_functions import count, no_count, primitive_feature, evaluation, \
    multiple_mutator, invalid_tree
from gencome.grapher import visualize_individual
from gencome.utils import list_dfs_to_tree, get_decision_rules, str_individual_with_real_feature_names
from gencome.file_utils import save_rules


# This part is outside of the "if main" check because of DEAP's issues with multiprocessing on Windows.
# If registering the multiprocessing map is moved inside the "if main" check it will fail.

parser = argparse.ArgumentParser(description=description)

parser.add_argument("--x_file_path",
                    help="a path to an input csv file; it has to have an id column with"
                    " string value that uniquely identifies a given object (the same id "
                    " has to be used in the corresponding y csv file), and in the remaining columns "
                    " the features describing a given object. Note that usually there will be many entries "
                    " for each object (e.g., an object could be a source file while all entries with that id "
                    " would be lines in that file).", 
                    type=str, required=True)

parser.add_argument("--y_file_path",
                    help="a path to an output csv file; it has to have at least two columsn: id and value, "
                    " the ids have to be the same as the ones used in the x file, while the value "
                    " column has to contain the numbers that the counted objects in the x file should "
                    " correlate with (e.g., entries in x could be lines of code while the values in y "
                    " could be the development effort for each file used as id).", 
                    type=str, required=True)

parser.add_argument("--results_dir_path",
                    help="a path to a directory where the results will be stored.", 
                    type=str, default="./results")

parser.add_argument("--sep",
                    help="a csv file separator.", 
                    type=str, default="$")

parser.add_argument("--correlation",
                    help="a correlation coefficient to be used when evaluating individuals.", 
                    type=str, choices=["Spearman", "Pearson"], default="Spearman")

parser.add_argument("--threads",
                    help="a number of threads to be used (default is the number of max(#CPUs-1, 1).", 
                    type=int, default=max(multiprocessing.cpu_count()-1, 1))

parser.add_argument("--min_tree_depth",
                    help="a minimum depth of a tree with the metric definition.", 
                    type=int, default=1)

parser.add_argument("--max_tree_depth",
                    help="a maximum depth of a tree with the metric definition.", 
                    type=int, default=3)

parser.add_argument("--tournament_size",
                    help="a number of individuals to take the best in each tournament.", 
                    type=int, default=5)

parser.add_argument("--population_size",
                    help="a number of individuals in the starting population.", 
                    type=int, default=50)

parser.add_argument("--crossover_prob",
                    help="a probability for performing crossover operation.", 
                    type=float, default=0.9)

parser.add_argument("--mutate_prob",
                    help="a probability for the mutation to appear.", 
                    type=float, default=0.3)

parser.add_argument("--generations",
                    help="a number of generations to perform.", 
                    type=int, default=1)

parser.add_argument("--mut_uniform_weight",
                    help="a weight for the chance of perfoming a uniform mutation.", 
                    type=int, default=1)

parser.add_argument("--mut_replacement_weight",
                    help="a weight for the chance of perfoming a replacement mutation.", 
                    type=int, default=1)

parser.add_argument("--mut_insert_weight",
                    help="a weight for the chance of perfoming an insert mutation.", 
                    type=int, default=1)

parser.add_argument("--mut_shrink_weight",
                    help="a weight for the chance of perfoming a shrink mutation.", 
                    type=int, default=1)

parser.add_argument("--top_best",
                    help="a number of the 'best' individuals to report.", 
                    type=int, default=10)

args = vars(parser.parse_args())
print(f"Run parameters: {str(args)}")


x_file_path = args['x_file_path']
y_file_path = args['y_file_path']
results_dir_path = args['results_dir_path']
sep = args['sep']
threads = args['threads']
top_best = args['top_best']
gencome.config.correlation = args['correlation']
gencome.config.min_tree_depth = args['min_tree_depth']
gencome.config.max_tree_depth = args['max_tree_depth']
gencome.config.tournament_size = args['tournament_size']
gencome.config.population_size = args['population_size']
gencome.config.crossover_prob = args['crossover_prob']
gencome.config.mutate_prob = args['mutate_prob']
gencome.config.generations = args['generations']
gencome.config.mut_uniform_weight = args['mut_uniform_weight']
gencome.config.mut_replacement_weight = args['mut_replacement_weight']
gencome.config.mut_insert_weight = args['mut_insert_weight']
gencome.config.mut_shrink_weight = args['mut_shrink_weight']

if not os.path.isfile(x_file_path):
    print(f"{x_file_path} doesn't exist")
    exit(1)

if not os.path.isfile(y_file_path):
    print(f"{y_file_path} doesn't exist")
    exit(1)

if not os.path.isdir(results_dir_path):
    os.makedirs(results_dir_path)

#Load data
x_file = pd.read_csv(x_file_path, sep=sep)
x_grouped = [x for x in x_file.groupby('id')]
x_features = {index : features.drop(["id"], axis=1).values for index, features in x_grouped}
gencome.config.features = x_file.drop(["id"], axis=1).columns.tolist()


y_file = pd.read_csv(y_file_path, sep=sep)
y_file = y_file
gencome.config.y = {y[1]['id']:y[1]['value'] for y in y_file[['id', 'value']].iterrows()}


decision_set = gp.PrimitiveSet(name="BaseSet", arity=0)
for index, feature in enumerate(gencome.config.features, start=0):
    decision_set.addPrimitive(primitive_feature(feature), arity=2, 
                    name=gencome.config.BASE_FEATURE_NAME + str(index))
decision_set.addTerminal(count, name=gencome.config.COUNT_LABEL)
decision_set.addTerminal(no_count, name=gencome.config.NOT_COUNT_LABEL)

# GP declaration
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=decision_set)

toolbox = base.Toolbox()
gencome.config.toolbox = toolbox

# Attribute generator
toolbox.register("expr", gp.genGrow, pset=decision_set, 
        min_=gencome.config.min_tree_depth, max_=gencome.config.max_tree_depth)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluation, x_features, pset=decision_set)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, 
            min_=gencome.config.min_tree_depth, max_=gencome.config.max_tree_depth)
toolbox.register("mutate", multiple_mutator, pset=decision_set)
toolbox.register("select", tools.selTournament, tournsize=gencome.config.tournament_size)

toolbox.decorate("mutate", invalid_tree())
toolbox.decorate("mate", invalid_tree())

if __name__ == '__main__':
    pool = multiprocessing.Pool(threads)
    toolbox.register("map", pool.map)

    population = toolbox.population(n=gencome.config.population_size)
    hof = tools.HallOfFame(top_best)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(population, toolbox, gencome.config.crossover_prob, 
            gencome.config.mutate_prob, gencome.config.generations, 
            stats=stats, halloffame=hof)

    pool.close()
    pool.join()

    with open(os.path.join(results_dir_path, "logbook.pickle"), 'wb') as f:
        pickle.dump(logbook, f)

    with open(os.path.join(results_dir_path, "rules.txt"), 'w') as f:
        for i, hof_ind in enumerate(hof):
            head_node = list_dfs_to_tree(hof_ind)
            print(f"Definition {i+1}: {str_individual_with_real_feature_names(hof_ind)}")
            graph = visualize_individual(head_node)
            graph.draw(path=os.path.join(results_dir_path, f"tree-top-{str(i+1)}.png"),
                    format="png", prog="dot")
            graph.draw(path=os.path.join(results_dir_path, f"tree-top-{str(i+1)}.pdf"),
                    format="pdf", prog="dot")
            graph.draw(path=os.path.join(results_dir_path, f"tree-top-{str(i+1)}.dot"),
                    format="dot", prog="dot")
            save_rules(f, get_decision_rules(hof_ind), i+1, hof_ind)

