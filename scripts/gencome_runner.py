
description="""Runs the process of finding count-based measures."""

import sys
import os
import numpy as np
import pandas as pd
import argparse
import operator
import pickle
from timeit import default_timer as timer
import random
import json

import copy
from collections import defaultdict, OrderedDict
import gc

from deap import algorithms, creator, base, gp, tools

import multiprocessing

import gencome.config 
from gencome.gp_functions import count, no_count, primitive_feature, evaluation, \
    multiple_mutator, invalid_tree, gen_grow
from gencome.grapher import visualize_individual
from gencome.utils import list_dfs_to_tree, get_decision_rules, str_individual_with_real_feature_names
from gencome.file_utils import save_rules

logger = gencome.config.logger

# The arguments are loaded outside of the "if main" check so workers have access to runtime settings

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

parser.add_argument("--fitness_type",
                    help="a correlation coefficient to be used when evaluating individuals.", 
                    type=str, choices=["FitnessMax", "FitnessMin"], default="FitnessMax")

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

parser.add_argument("--random_state",
                    help="a random seed.", 
                    type=int, default=24110)

args = vars(parser.parse_args())
print(f"Run parameters: {str(args)}")


x_file_path = args['x_file_path']
y_file_path = args['y_file_path']
results_dir_path = args['results_dir_path']
sep = args['sep']
threads = args['threads']
top_best = args['top_best']
random_state = args['random_state']
gencome.config.correlation = args['correlation']
gencome.config.fitness_type = args['fitness_type']
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

# Each worker need to initialize individual so it 
# needs to load minimum data from the input file to do so.
if multiprocessing.current_process().name != "MainProcess":
    logger.debug(f"Loading data in the worker: {multiprocessing.current_process().name}")
    start = timer()
    x_file = next(pd.read_csv(x_file_path, sep=sep, chunksize=1))
    end = timer()
    logger.debug(f"Loaded data in the worker: {multiprocessing.current_process().name} ({end-start:.2f}s)...")

    id_index = x_file.columns.tolist().index("id")
    columns = x_file.columns.tolist()
    del columns[id_index]
    gencome.config.features = columns

    decision_set = gp.PrimitiveSet(name="BaseSet", arity=0)
    for index, feature in enumerate(gencome.config.features, start=0):
        decision_set.addPrimitive(primitive_feature(feature), arity=2, 
                        name=gencome.config.BASE_FEATURE_NAME + str(index))
    decision_set.addTerminal(count, name=gencome.config.COUNT_LABEL)
    decision_set.addTerminal(no_count, name=gencome.config.NOT_COUNT_LABEL)

    # GP declaration
    if gencome.config.fitness_type == "FitnessMax":
        creator.create(gencome.config.fitness_type, base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=decision_set)
    else:
        creator.create(gencome.config.fitness_type, base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=decision_set)
    
    logger.debug(f"Worker {multiprocessing.current_process().name} ready!")

if __name__ == '__main__':

    #Load data
    logger.debug(f"Loading X data from {x_file_path}...")
    start = timer()
    x_file = pd.read_csv(x_file_path, sep=sep)
    end = timer()
    logger.debug(f"Loaded X data from {x_file_path} ({end-start:.2f}s)...")

    logger.debug("Grouping X data by id...")
    start = timer()
    x_groups = defaultdict(lambda: [])
    id_index = x_file.columns.tolist().index("id")
    columns = x_file.columns.tolist()
    del columns[id_index]
    x_features = OrderedDict()
    for i, row in enumerate(x_file.values):
        if i % 10000 == 0:
            logger.debug(f"Processing {i+1:,} row from X...")
        new_row = row
        x_groups[str(new_row[id_index])].append(tuple(np.delete(new_row, id_index).tolist()))
    logger.debug("Grouping by id...")
    for key in x_groups.keys():
        if i % 1000 == 0:
            logger.debug(f"Grouping by {i+1:,} id '{key}' from X ...")
        x_features[key] = pd.DataFrame(x_groups.get(key)).values
    del x_groups
    gencome.config.features = columns
    gencome.config.x_features = x_features
    end = timer()
    logger.debug(f"Finished the grouping of the X data by id ({end-start:.2f}s)...")
    gc.collect()

    logger.debug(f"Loading Y data from {x_file_path}...")
    y_file = pd.read_csv(y_file_path, sep=sep)
    gencome.config.y_dict = {y[1]['id']:y[1]['value'] for y in y_file[['id', 'value']].iterrows()}
    gencome.config.y = [gencome.config.y_dict[index] for index in x_features]

    logger.debug("Configuring the toolbox...")
    decision_set = gp.PrimitiveSet(name="BaseSet", arity=0)
    for index, feature in enumerate(gencome.config.features, start=0):
        decision_set.addPrimitive(primitive_feature(feature), arity=2, 
                        name=gencome.config.BASE_FEATURE_NAME + str(index))
    decision_set.addTerminal(count, name=gencome.config.COUNT_LABEL)
    decision_set.addTerminal(no_count, name=gencome.config.NOT_COUNT_LABEL)

    # GP declaration
    if gencome.config.fitness_type == "FitnessMax":
        creator.create(gencome.config.fitness_type, base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=decision_set)
    else:
        creator.create(gencome.config.fitness_type, base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=decision_set)

    toolbox = base.Toolbox()
    gencome.config.toolbox = toolbox

    # Attribute generator
    toolbox.register("expr", gen_grow, pset=decision_set, 
            min_=gencome.config.min_tree_depth, max_=gencome.config.max_tree_depth)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    data = (gencome.config.max_tree_depth, gencome.config.x_features, gencome.config.y)
    toolbox.register("evaluate", evaluation, data, pset=decision_set)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gen_grow, 
                min_=gencome.config.min_tree_depth, max_=gencome.config.max_tree_depth)
    toolbox.register("mutate", multiple_mutator, pset=decision_set)
    toolbox.register("select", tools.selTournament, tournsize=gencome.config.tournament_size)

    toolbox.decorate("mutate", invalid_tree())
    toolbox.decorate("mate", invalid_tree())

    random.seed(random_state)
    np.random.seed(random_state)

    pool = multiprocessing.Pool(threads)
    toolbox.register("map", pool.map)

    population = toolbox.population(n=gencome.config.population_size)
    hof = tools.HallOfFame(top_best)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logger.debug("Running the algorithm...")
    population, logbook = algorithms.eaSimple(population, toolbox, gencome.config.crossover_prob, 
            gencome.config.mutate_prob, gencome.config.generations, 
            stats=stats, halloffame=hof)

    population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True) 

    pool.close()
    pool.join()

    logger.debug("Reporting...")
    with open(os.path.join(results_dir_path, "logbook.pickle"), 'wb') as f:
        pickle.dump(logbook, f)
    
    with open(os.path.join(results_dir_path, "rules.txt"), 'w') as f:
        for i, hof_ind in enumerate(hof):
            head_node = list_dfs_to_tree(hof_ind)
            print(f"Definition Top#{i+1}: {str_individual_with_real_feature_names(hof_ind)}")
            graph = visualize_individual(head_node)
            try:
                graph.draw(path=os.path.join(results_dir_path, f"tree-top-{str(i+1)}.png"),
                        format="png", prog="dot")
            except: 
                print(f"Unable to generate tree-top-{str(i+1)}.png")
            try:
                graph.draw(path=os.path.join(results_dir_path, f"tree-top-{str(i+1)}.pdf"),
                        format="pdf", prog="dot")
            except: 
                print(f"Unable to generate tree-top-{str(i+1)}.pdf")
            try:
                graph.draw(path=os.path.join(results_dir_path, f"tree-top-{str(i+1)}.dot"),
                        format="dot", prog="dot")
            except: 
                print(f"Unable to generate tree-top-{str(i+1)}.dot")
            save_rules(f, get_decision_rules(hof_ind), f"Top#{i+1}", hof_ind)

        for j, pop_ind in enumerate(population):
            if pop_ind not in hof:
                print(f"Definition Pop#{j+1}: {str_individual_with_real_feature_names(pop_ind)}")
                head_node = list_dfs_to_tree(pop_ind)
                graph = visualize_individual(head_node)
                try:
                    graph.draw(path=os.path.join(results_dir_path, f"tree-pop-{str(j+1)}.png"),
                            format="png", prog="dot")
                except: 
                    print(f"Unable to generate tree-pop-{str(j+1)}.png")
                try:
                    graph.draw(path=os.path.join(results_dir_path, f"tree-pop-{str(j+1)}.pdf"),
                            format="pdf", prog="dot")
                except: 
                    print(f"Unable to generate tree-pop-{str(j+1)}.pdf")
                try:
                    graph.draw(path=os.path.join(results_dir_path, f"tree-pop-{str(j+1)}.dot"),
                            format="dot", prog="dot")
                except: 
                    print(f"Unable to generate tree-pop-{str(j+1)}.dot")
                save_rules(f, get_decision_rules(pop_ind), f"Pop#{j+1}", pop_ind)

    individuals = []
    for i, ind in enumerate(hof):
        individuals.append((f"Top#{i+1}", str(ind), str_individual_with_real_feature_names(ind), ind.fitness.values[0]))
    for i, ind in enumerate(population):
        if ind not in hof:
            individuals.append((f"Pop#{i+1}", str_individual_with_real_feature_names(ind), ind.fitness.values[0]))   
    with open(os.path.join(results_dir_path, "trees.json"), 'w', encoding='utf-8') as fj:
        json.dump(individuals, fj, indent=4, ensure_ascii=True)

