import sys
import numpy as np
import math
from deap import algorithms, creator, base, gp, tools

from scripts.constants import BASE_FEATURE_NAME, MAX_TREE_DEPTH, \
    MIN_TREE_DEPTH, TOURNAMENT_SIZE, POPULATION_SIZE, CROSS_PROBABILITY, MUTATE_PROBABILITY, GENERATION_COUNT
from scripts.gp_functions import set_features, count, no_count, primitive_feature
from scripts.grapher import visualize_individual
from scripts.file_utils import load_data
from scripts.utils import list_dfs_to_tree


def evaluation(data, individual, pset):
    func = gp.compile(expr=individual, pset=pset)
    class_count_result = []
    for class_name in data:
        lines_count = 0
        for line_number in data[class_name]:
            if func(data[class_name][line_number]):
                lines_count += 1
        class_count_result.append(lines_count)
    eval_result = 0
    for i, (test_class_name, test_class_value) in enumerate(data_holder.test_data['code'].items()):
        eval_result += abs(int(test_class_value) - class_count_result[i])
    # return eval_result,
    value = np.corrcoef(class_count_result, data_holder.test_data_values['code']).min()
    if math.isnan(value):
        return 0,
    return value,


if len(sys.argv) == 1:
    print("No argument for data folder path")
    exit()

data_file_path = sys.argv[1]

if len(data_file_path) < 1:
    print("data_file_path was empty")
    exit()

#Load data
data_holder = load_data(data_file_path)


set_features(data_holder.features)

# GP declaration
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Tree", gp.PrimitiveTree, fitness=creator.FitnessMax)

decision_set = gp.PrimitiveSet(name="BaseSet", arity=0)
for index, feature in enumerate(data_holder.features, start=0):
    decision_set.addPrimitive(primitive_feature(feature), arity=2, name=BASE_FEATURE_NAME + str(index))
decision_set.addTerminal(count, name="Count")
decision_set.addTerminal(no_count, name="NoCount")


toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr", gp.genGrow, pset=decision_set, min_=MIN_TREE_DEPTH, max_=MAX_TREE_DEPTH)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Tree, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluation, data_holder.features_data, pset=decision_set)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=MIN_TREE_DEPTH, max_=MAX_TREE_DEPTH)
toolbox.register("mutate", gp.mutNodeReplacement, pset=decision_set)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

population = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1)

algorithms.eaSimple(population, toolbox, CROSS_PROBABILITY, MUTATE_PROBABILITY, GENERATION_COUNT, stats=stats, halloffame=hof)

head_node = list_dfs_to_tree(hof[0])
print(hof[0].fitness)

visualize_individual(data_holder, head_node)
