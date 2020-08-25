
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
import logging

import copy
from collections import defaultdict, OrderedDict
import gc

from deap import algorithms, creator, base, gp, tools

import multiprocessing

import gencome.config 
from gencome.gp_functions import count, no_count, primitive_feature, evaluation, \
    multiple_mutator, invalid_tree, gen_grow
from gencome.grapher import visualize_individual
from gencome.utils import list_dfs_to_tree, get_decision_rules, str_individual_with_real_feature_names, \
                        summarize_individual
from gencome.file_utils import save_rules

logger = gencome.config.logger

# The arguments are loaded outside of the "if main" check so workers have access to runtime settings

parser = argparse.ArgumentParser(description=description)

parser.add_argument("--x_file_path",
                    help="paths to input csv files; they have to have an id column with"
                    " string value that uniquely identifies a given object (the same id "
                    " has to be used in the corresponding y csv file), and in the remaining columns "
                    " the features describing a given object. Note that usually there will be many entries "
                    " for each object (e.g., an object could be a source file while all entries with that id "
                    " would be lines in that file).", 
                    type=str, nargs='+', required=True)

parser.add_argument("--y_file_path",
                    help="paths to output csv files; they have to have at least two columns: id and value, "
                    " the ids have to be the same as the ones used in the x file, while the value "
                    " column has to contain the numbers that the counted objects in the x file should "
                    " correlate with (e.g., entries in x could be lines of code while the values in y "
                    " could be the development effort for each file used as id).", 
                    type=str, nargs='+', required=True)

parser.add_argument("--results_dir_path",
                    help="a path to a directory where the results will be stored.", 
                    type=str, default="./results")

parser.add_argument("--sep",
                    help="a csv file separator.", 
                    type=str, default="$")

parser.add_argument("--correlation",
                    help="a correlation coefficient to be used when evaluating individuals.", 
                    type=str, choices=["Spearman", "Pearson", "Kendall"], default="Kendall")

parser.add_argument("--fitness_type",
                    help="a correlation coefficient to be used when evaluating individuals.", 
                    type=str, nargs='+', choices=["max", "min"], default=["max"])

parser.add_argument("--threads",
                    help="a number of threads to be used (default is the number of max(#CPUs-1, 1).", 
                    type=int, default=max(multiprocessing.cpu_count()-1, 1))

parser.add_argument("--min_tree_depth",
                    help="a minimum depth of a tree with the metric definition.", 
                    type=int, default=1)

parser.add_argument("--max_tree_depth",
                    help="a maximum depth of a tree with the metric definition.", 
                    type=int, default=3)

parser.add_argument("--min_init_tree_depth",
                    help="a minimum depth of a tree during the individuals generation.", 
                    type=int, default=0)

parser.add_argument("--max_init_tree_depth",
                    help="a maximum depth of a tree during the individuals generation.", 
                    type=int, default=2)

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

parser.add_argument("--logging_level",
                    help="a level of verbosity.", 
                    type=str, choices=["DEBUG", "INFO"], default="INFO")

args = vars(parser.parse_args())
print(f"Run parameters: {str(args)}")


x_file_paths = args['x_file_path']
y_file_paths = args['y_file_path']
results_dir_path = args['results_dir_path']
sep = args['sep']
threads = args['threads']
top_best = args['top_best']
random_state = args['random_state']
gencome.config.correlation = args['correlation']
gencome.config.fitness_type = args['fitness_type']
gencome.config.min_tree_depth = args['min_tree_depth']
gencome.config.max_tree_depth = args['max_tree_depth']
gencome.config.min_init_tree_depth = args['min_init_tree_depth']
gencome.config.max_init_tree_depth = args['max_init_tree_depth']
gencome.config.tournament_size = args['tournament_size']
gencome.config.population_size = args['population_size']
gencome.config.crossover_prob = args['crossover_prob']
gencome.config.mutate_prob = args['mutate_prob']
gencome.config.generations = args['generations']
gencome.config.mut_uniform_weight = args['mut_uniform_weight']
gencome.config.mut_replacement_weight = args['mut_replacement_weight']
gencome.config.mut_insert_weight = args['mut_insert_weight']
gencome.config.mut_shrink_weight = args['mut_shrink_weight']

if args['logging_level'] == "DEBUG":
    gencome.config.logger.setLevel(logging.DEBUG)
    gencome.config.ch.setLevel(logging.DEBUG)

for x_file_path in x_file_paths:
    if not os.path.isfile(x_file_path):
        print(f"{x_file_path} doesn't exist")
        exit(1)

for y_file_path in y_file_paths:
    if not os.path.isfile(y_file_path):
        print(f"{y_file_path} doesn't exist")
        exit(1)

if not os.path.isdir(results_dir_path):
    os.makedirs(results_dir_path)

# Each worker need to initialize individual so it 
# needs to load minimum data from the input file to do so.
if multiprocessing.current_process().name != "MainProcess":

    common_columns = None
    x_files = []
    for x_file_path in x_file_paths: 
        logger.debug(f"Loading data in the worker: {multiprocessing.current_process().name}")
        start = timer()
        x_file = next(pd.read_csv(x_file_path, sep=sep, chunksize=1))
        x_files.append(x_file)
        end = timer()
        logger.debug(f"Loaded data in the worker: {multiprocessing.current_process().name} ({end-start:.2f}s)...")

        columns = x_file.columns.tolist()
        if common_columns is None:
            common_columns = set(columns)
        else:
            common_columns = common_columns.intersection(set(columns))

    common_columns = list(common_columns)
    id_index = common_columns.index("id")
    del common_columns[id_index]
    common_columns.sort()
    gencome.config.features = common_columns

    decision_set = gp.PrimitiveSet(name="BaseSet", arity=0)
    for index, feature in enumerate(gencome.config.features, start=0):
        decision_set.addPrimitive(primitive_feature(feature), arity=2, 
                        name=gencome.config.BASE_FEATURE_NAME + str(index))
    decision_set.addTerminal(count, name=gencome.config.COUNT_LABEL)
    decision_set.addTerminal(no_count, name=gencome.config.NOT_COUNT_LABEL)

    fitness_weights = [1.0 if fitness_type == 'max' else -1.0 for fitness_type in gencome.config.fitness_type]
    # GP declaration
    creator.create("Fitness", base.Fitness, weights=fitness_weights)
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, pset=decision_set)
    
    logger.debug(f"Worker {multiprocessing.current_process().name} ready!")

if __name__ == '__main__':

    entire_process_start = timer()

    #Load data
    common_columns = None
    x_files = []
    for x_file_path in x_file_paths:
        logger.debug(f"Loading X data from {x_file_path}...")
        start = timer()
        logger.debug(f"Probabing data types in X data...")
        x_file = next(pd.read_csv(x_file_path, sep=sep, chunksize=1))
        d = dict.fromkeys(x_file.select_dtypes([np.int64, np.float64, np.int32, np.int16]).columns, np.uint8)
        logger.debug(f"Loading X data...")
        x_file = pd.read_csv(x_file_path, sep=sep, dtype=d)
        #x_file = pd.read_csv(x_file_path, sep=sep)
        #d = dict.fromkeys(x_file.select_dtypes([np.int64, np.float64, np.int32, np.int16]).columns, np.uint8)
        #x_file = x_file.astype(d)
        x_files.append(x_file)
        end = timer()
        logger.debug(f"Loaded X data from {x_file_path} ({end-start:.2f}s)...")

        columns = x_file.columns.tolist()
        if common_columns is None:
            common_columns = set(columns)
        else:
            common_columns = common_columns.intersection(set(columns))
    
    common_columns = list(common_columns)
    id_index = common_columns.index("id")
    del common_columns[id_index]
    common_columns.sort()
    gencome.config.features = common_columns

    
    x_files_unified_cols = []
    for x_file in x_files:
        x_files_unified_cols.append(x_file[['id']+common_columns])
        del x_file
    x_files = x_files_unified_cols
    gc.collect()

    logger.debug("Grouping X data by id...")
    start = timer()
    x_features = []
    for x_file in x_files:
        id_index = x_file.columns.tolist().index('id')
        x_groups = defaultdict(lambda: [])
        x_file_features = OrderedDict()
        for i, row in enumerate(x_file.values):
            if i % 10000 == 0:
                logger.debug(f"Processing {i+1:,} row from X...")
            new_row = row
            x_groups[str(new_row[id_index])].append(tuple(np.delete(new_row, id_index).tolist()))
        logger.debug("Grouping by id...")
        for key in x_groups.keys():
            if i % 1000 == 0:
                logger.debug(f"Grouping by {i+1:,} id '{key}' from X ...")
            x_file_features[key] = pd.DataFrame(x_groups.get(key)).values
        del x_groups
        x_features.append(x_file_features)
    gencome.config.x_features = x_features
    
    end = timer()
    logger.debug(f"Finished the grouping of the X data by id ({end-start:.2f}s)...")
    gc.collect()

    gencome.config.y = []
    for i, y_file_path in enumerate(y_file_paths):
        logger.debug(f"Loading Y data from {y_file_path}...")
        y_file = pd.read_csv(y_file_path, sep=sep)
        y_dict = {y[1]['id']:y[1]['value'] for y in y_file[['id', 'value']].iterrows()}
        gencome.config.y_dicts.append(y_dict)
        gencome.config.y.append([y_dict[index] for index in x_features[i]])

    logger.debug("Configuring the toolbox...")
    decision_set = gp.PrimitiveSet(name="BaseSet", arity=0)
    for index, feature in enumerate(gencome.config.features, start=0):
        decision_set.addPrimitive(primitive_feature(feature), arity=2, 
                        name=gencome.config.BASE_FEATURE_NAME + str(index))
    decision_set.addTerminal(count, name=gencome.config.COUNT_LABEL)
    decision_set.addTerminal(no_count, name=gencome.config.NOT_COUNT_LABEL)


    fitness_weights = [1.0 if fitness_type == 'max' else -1.0 for fitness_type in gencome.config.fitness_type]
    # GP declaration
    creator.create("Fitness", base.Fitness, weights=fitness_weights)
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, pset=decision_set)

    toolbox = base.Toolbox()
    gencome.config.toolbox = toolbox

    # Attribute generator
    toolbox.register("expr", gen_grow, pset=decision_set, 
            min_=gencome.config.min_init_tree_depth, max_=gencome.config.max_init_tree_depth)

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

    if gencome.config.fitness_type[0] == "max":
        population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
    else:
        population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=False)

    pool.close()
    pool.join()

    logger.debug("Reporting...")
    with open(os.path.join(results_dir_path, "logbook.pickle"), 'wb') as f:
        pickle.dump(logbook, f)
    
    # store detailed information about the trees
    logger.debug("Saving information about trees...")
    individuals = []
    reported = set()
    for i, ind in enumerate(hof):
        individuals.append(summarize_individual(f"Top#{i+1}", ind))
        reported.add(str(ind))
    for i, ind in enumerate(population):
        if str(ind) not in reported:
            individuals.append(summarize_individual(f"Pop#{i+1}", ind))  
            reported.add(str(ind)) 
    logger.debug("Using json to store the information about the trees...")
    with open(os.path.join(results_dir_path, "trees.json"), 'w', encoding='utf-8') as fj:
        json.dump(individuals, fj, indent=4, ensure_ascii=True)

    # store compiled trees for further use
    logger.debug("Saving compiled trees...")
    individuals_compiled = {'features_names': gencome.config.features}
    individuals_compiled['features_map'] = { f"{gencome.config.BASE_FEATURE_NAME}{str(index)}": feature \
                        for index, feature in enumerate(gencome.config.features, start=0)}
    individuals_compiled['features_map_rev'] = { feature: f"{gencome.config.BASE_FEATURE_NAME}{str(index)}" \
                        for index, feature in enumerate(gencome.config.features, start=0)}
    individuals_compiled['trees'] = []
    reported = set()
    for i, ind in enumerate(hof):
        ind_dict = summarize_individual(f"Top#{i+1}", ind)
        ind_dict['compiled'] =  gp.compile(expr=ind, pset=decision_set)
        individuals_compiled['trees'].append(ind_dict)
        reported.add(str(ind))
    for i, ind in enumerate(population):
        if str(ind) not in reported:
            ind_dict = summarize_individual(f"Pop#{i+1}", ind)
            ind_dict['compiled'] =  gp.compile(expr=ind, pset=decision_set)
            individuals_compiled['trees'].append(ind_dict)
            reported.add(str(ind))
    logger.debug("Using pickle to store the trees...")
    with open(os.path.join(results_dir_path, "trees.pickle"), 'wb') as fp:
        pickle.dump(individuals_compiled, fp)
    
    logger.debug("Saving the rules...")
    with open(os.path.join(results_dir_path, "rules.txt"), 'w') as f:
        reported = set()
        for i, hof_ind in enumerate(hof):
            reported.add(str(hof_ind)) 
            print(f"Definition Top#{i+1}: {str_individual_with_real_feature_names(hof_ind)}")
            save_rules(f, get_decision_rules(hof_ind), f"Top#{i+1}", hof_ind)
        
        for j, pop_ind in enumerate(population):
            if str(pop_ind) not in reported:
                reported.add(str(pop_ind))
                print(f"Definition Pop#{j+1}: {str_individual_with_real_feature_names(pop_ind)}")
                save_rules(f, get_decision_rules(pop_ind), f"Pop#{j+1}", pop_ind)
    
    logger.debug("Saving graphical representation of the trees...")
    reported = set()
    for i, hof_ind in enumerate(hof):
        reported.add(str(hof_ind)) 
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
    
    for j, pop_ind in enumerate(population):
        if str(pop_ind) not in reported:
            reported.add(str(pop_ind))
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
    
    entire_process_end = timer()
    logger.debug(f"Finished in {entire_process_end-entire_process_start:.2f}s ...")
