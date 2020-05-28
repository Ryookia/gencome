from scripts.node import Node


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


def print_individual(individual, features, base_feature_name):
    for node_index in range(len(individual)):
        node = individual[node_index]
        if base_feature_name in node.name:
            print(node.name + " " + features[int(node.name.replace(base_feature_name, ''))])
        else:
            print(node.name)
