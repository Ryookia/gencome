import sys
import csv
import os
import numpy as np
from deap import algorithms, creator, base, gp, tools
from functools import partial

FEATURES_FILE = "/train-features.csv"
CLASS_KEYWORD = "class"
TEST_KEYWORD = "test"
FILE_SEPARATOR = "_"

MAX_TREE_DEPTH = 5
MIN_TREE_DEPTH = 1

POPULATION_SIZE = 100

TOURNAMENT_SIZE = 3


def evaluation(data, individual, pset):
    func = gp.compile(expr=individual, pset=pset)
    class_count_result = []
    for class_name in data:
        for line_number in data[class_name]:
            class_count_result.append(func(data[class_name][line_number]))
    return np.corrcoef(class_count_result, test_data['code'])


def has_keyword(keyword, line_features):
    return line_features[features.index(keyword)]


def primitive_keyword(keyword):
    return partial(has_keyword, keyword)


if len(sys.argv) == 1:
    print("No argument for data folder path")
    exit()

features = []
# Dictionary of class files with dictionary for each line
features_data = {}
class_data = {}
test_data = {}
value_types = []
data_file_path = sys.argv[1]

#Load data

with open(data_file_path + FEATURES_FILE) as csvFile:
    reader = csv.reader(csvFile, delimiter='$')
    features = next(reader)
    current_file = None
    for line in reader:
        file_name = line[0].split(":")[0]
        file_line = line[0].split(":")[1]
        if current_file != file_name:
            features_data[file_name] = {}
            current_file = file_name
        features_data[file_name][int(file_line)] = line[1:]

file_names = os.listdir(data_file_path)

for file_name in file_names:
    current_file_data = file_name.split(FILE_SEPARATOR)
    if len(current_file_data) == 1:
        continue
    if current_file_data[0] != CLASS_KEYWORD and current_file_data[0] != TEST_KEYWORD:
        continue
    value_type = current_file_data[1].split(".")[0]
    value_types.append(value_type)
    if current_file_data[0] == CLASS_KEYWORD:
        class_data[value_type] = {}
        with open(data_file_path + "/" + file_name) as csvFile:
            reader = csv.reader(csvFile, delimiter='$')
            next(reader)
            for line in reader:
                class_data[value_type][line[1]] = line[2]
    if current_file_data[0] == TEST_KEYWORD:
        test_data[value_type] = {}
        with open(data_file_path + "/" + file_name) as csvFile:
            reader = csv.reader(csvFile, delimiter='$')
            next(reader)
            for line in reader:
                test_data[value_type][line[1]] = line[2]

# GP declaration
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Tree", gp.PrimitiveTree, fitness=creator.FitnessMax)

decision_set = gp.PrimitiveSet(name="BaseSet", arity=1)
for index, feature in enumerate(features, start=0):
    decision_set.addPrimitive(primitive_keyword(feature), arity=1, name="Feature" + str(index))


toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr", gp.genFull, pset=decision_set, min_=MIN_TREE_DEPTH, max_=MAX_TREE_DEPTH)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Tree, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluation, features_data, pset=decision_set)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.mutUniform, min=0, max=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=decision_set)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

population = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1)

algorithms.eaSimple(population, toolbox, 0.5, 0.2, 40, halloffame=hof)

print(hof)


