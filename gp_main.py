import sys
import csv
import os
import numpy as np
from deap import algorithms, creator, base, gp, tools
from functools import partial
import pygraphviz as pgv
from node import Node

FEATURES_FILE = "/train-features.csv"
CLASS_KEYWORD = "class"
TEST_KEYWORD = "test"
FILE_SEPARATOR = "_"

MAX_TREE_DEPTH = 2
MIN_TREE_DEPTH = 1

POPULATION_SIZE = 100

TOURNAMENT_SIZE = 3

CROSS_PROBABILITY = 0.5
MUTATE_PROBABILITY = 0.2
GENERATION_COUNT = 5

BASE_FEATURE_NAME = "Feature"


def list_dfs_to_tree(val_list):
    head = Node()
    current_index = 0
    head.value = val_list[current_index]
    if head.value.arity == 0:
        return head
    current_index += 1
    head.leftChild = Node()
    head.leftChild.value = val_list[current_index]
    current_index = add_node(val_list, current_index, head.leftChild)
    current_index += 1
    head.rightChild = Node()
    head.rightChild.value = val_list[current_index]
    add_node(val_list, current_index, head.rightChild)
    return head


def add_node(val_list, current_index, node):
    if node.value.arity == 0:
        return current_index
    current_index += 1
    node.leftChild = Node()
    node.leftChild.value = val_list[current_index]
    current_index = add_node(val_list, current_index, node.leftChild)
    current_index += 1
    node.rightChild = Node()
    node.rightChild.value = val_list[current_index]
    return add_node(val_list, current_index, node.rightChild)


def print_individual(individual):
    for node_index in range(len(individual)):
        node = individual[node_index]
        if BASE_FEATURE_NAME in node.name:
            print(node.name + " " + features[int(node.name.replace(BASE_FEATURE_NAME, ''))])
        else:
            print(node.name)


def visualize_individual(head_node):
    graph = pgv.AGraph(strict=False, directed=True)
    add_node_to_graph(graph, head_node)
    graph.layout()
    graph.draw('file2.png')
    print(graph)


def add_node_to_graph(graph, node):
    if node.value.arity == 0:
        return
    graph.add_edge(node, node.rightChild)
    # graph.add_edge(str(node) + " " + node.value.name, str(node.rightChild) + " " + node.rightChild.value.name)
    # graph.add_edge(str(node) + " " + node.value.name, str(node.leftChild) + " " + node.leftChild.value.name)
    graph.add_edge(node, node.leftChild)
    graph.get_node(node).attr["label"] = node.value.name
    graph.get_node(node.rightChild).attr["label"] = node.rightChild.value.name
    graph.get_node(node.leftChild).attr["label"] = node.leftChild.value.name
    add_node_to_graph(graph, node.leftChild)
    add_node_to_graph(graph, node.rightChild)


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
    for i, (test_class_name, test_class_value) in enumerate(test_data['code'].items()):
        eval_result += abs(int(test_class_value) - class_count_result[i])
    # return eval_result,

    return np.corrcoef(class_count_result, test_data_values['code']).min(),


#should not pass line_features to out
def has_keyword(keyword, out1, out2, line_features):
    if (int(line_features[features.index(keyword)])) > 0:
        return out1(line_features)
    else:
        return out2(line_features)


def count(_):
    return True


def no_count(_):
    return False


def primitive_feature(keyword):
    return partial(primitive_keyword, keyword)


def primitive_keyword(keyword, out1, out2):
    return partial(has_keyword, keyword, out1, out2)


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
test_data_values = {}

if len(data_file_path) < 1:
    print("data_file_path was empty")
    exit()

#Load data

with open(data_file_path + FEATURES_FILE) as csvFile:
    reader = csv.reader(csvFile, delimiter='$')
    features = next(reader)[1:-1]
    current_file = None
    for line in reader:
        file_name = line[0].split(":")[0]
        file_line = line[0].split(":")[1]
        if current_file != file_name:
            features_data[file_name] = {}
            current_file = file_name
        features_data[file_name][int(file_line)] = line[1:-1]

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

for type_key in test_data.keys():
    if type_key not in test_data_values:
        test_data_values[type_key] = []
        for key in test_data[type_key]:
            test_data_values[type_key].append(int(test_data[type_key][key]))

# GP declaration
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Tree", gp.PrimitiveTree, fitness=creator.FitnessMax)

decision_set = gp.PrimitiveSet(name="BaseSet", arity=0)
for index, feature in enumerate(features, start=0):
    decision_set.addPrimitive(primitive_feature(feature), arity=2, name=BASE_FEATURE_NAME + str(index))
decision_set.addTerminal(count, name="Count")
decision_set.addTerminal(no_count, name="NoCount")


toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr", gp.genFull, pset=decision_set, min_=MIN_TREE_DEPTH, max_=MAX_TREE_DEPTH)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Tree, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluation, features_data, pset=decision_set)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=decision_set)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

population = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1)

algorithms.eaSimple(population, toolbox, CROSS_PROBABILITY, MUTATE_PROBABILITY, GENERATION_COUNT, halloffame=hof)

print_individual(hof[0])
print("$$$$$$")
head_node = list_dfs_to_tree(hof[0])
print(head_node)
print("$$$$$$")

visualize_individual(head_node)
