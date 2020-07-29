from gencome.node import Node

import gencome.config


def str_individual_with_real_feature_names(individual):
    str_ind = str(individual)
    for index, feature in reversed(list(enumerate(gencome.config.features, start=0))):
                str_ind = str_ind.replace(gencome.config.BASE_FEATURE_NAME + str(index), f"'{feature}'")
    return str_ind

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


# TODO calculation is not valid
def calculate_tree_height(individual):
    if individual[0].arity == 0:
        return 1
    left_index, left_height = calculate_subtree_height(individual, 1, 2)
    right_index, right_height = calculate_subtree_height(individual, left_index, 2)
    if left_height > right_height:
        return left_height
    else:
        return right_height


def calculate_subtree_height(individual, index, height):
    if individual[index].arity == 0:
        return index, height
    left_index, left_height = calculate_subtree_height(individual, index + 1, height + 1)
    right_index, right_height = calculate_subtree_height(individual, left_index + 1, height + 1)
    if left_height > right_height:
        return left_height, right_index
    return right_height, right_index


def print_individual(individual, features, base_feature_name):
    for node_index in range(len(individual)):
        node = individual[node_index]
        if base_feature_name in node.name:
            print(node.name + " " + features[int(node.name.replace(base_feature_name, ''))])
        else:
            print(node.name)


def get_decision_rules(individual):
    if individual[0].arity == 0:
        return [({}, individual[0].name)]
    rules = []
    keyword_set = set()
    keyword_set.add((True, individual[0].name))
    index = 1
    index = get_rules_from_subtree(individual, index, keyword_set, rules)
    keyword_set = set()
    keyword_set.add((False, individual[0].name))
    get_rules_from_subtree(individual, index + 1, keyword_set.copy(), rules)
    remove_duplicate_rules(rules)
    return rules


def get_rules_from_subtree(individual, index, keyword_set, rules):
    if individual[index].arity == 0:
        rules.append((keyword_set, individual[index].name))
        return index
    left_set = keyword_set.copy()
    left_set.add((True, individual[index].name))
    right_set = keyword_set.copy()
    right_set.add((False, individual[index].name))
    index = get_rules_from_subtree(individual, index + 1, left_set, rules)
    index = get_rules_from_subtree(individual, index + 1, right_set, rules)
    return index


def get_primitive_keyword(name, features):
    return features[int(name.replace(gencome.config.BASE_FEATURE_NAME, ''))]


def remove_duplicate_rules(rules):
    duplicate_index = set()
    rules_count = len(rules)
    if rules_count <= 1:
        return
    for start, rule in enumerate(rules):
        if rules_count == start + 1:
            continue
        for index, next_rule in enumerate(rules[start + 1:]):
            if compare_rules(rule, next_rule):
                duplicate_index.add(index + start + 1)
    for index in sorted((list(duplicate_index)), reverse=True):
        rules.pop(index)


def compare_rules(rule1, rule2):
    if rule1[1] != rule2[1] or len(rule1[0]) != len(rule2[0]):
        return False
    return rule1[0] == rule2[0] and rule1[1] == rule2[1]

def summarize_individual(label, ind):
    count_or_rules = []
    rules = get_decision_rules(ind)
    for rule in rules:
        if rule[1] == gencome.config.COUNT_LABEL:
            # skup rules having the same keyword for true and false
            keywords_true = set()
            keywords_false = set()
            for keyword in rule[0]:
                if keyword[0]:
                   keywords_true.add(keyword[1]) 
                else:
                    keywords_false.add(keyword[1]) 
            if len(keywords_true.intersection(keywords_false)) > 0:
                continue
            keywords = []
            for keyword in rule[0]:
                if not keyword[0]:
                    keywords.append(gencome.config.NOT_PREFIX + "'"+get_primitive_keyword(keyword[1], gencome.config.features)+"'")
                else:
                    keywords.append("'"+get_primitive_keyword(keyword[1], gencome.config.features)+"'")
            count_or_rules.append(keywords)
    return {"label": label, 'depth': ind.height, 'raw_tree': str(ind), 
            'keywords_true_rules' : count_or_rules,
            'tree': str_individual_with_real_feature_names(ind), 
            'corr': ind.fitness.values[0], 'pvalue': ind.fitness.values[1]}
