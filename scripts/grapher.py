import pygraphviz as pgv
from scripts.constants import BASE_FEATURE_NAME

RED_COLOR = "#ff6666"
GREEN_COLOR = '#66ff99'

def visualize_individual(data_holder, head_node):
    graph = pgv.AGraph(strict=False, directed=True)
    add_node_to_graph(graph, head_node, data_holder)
    graph.layout(prog='dot')
    graph.draw('file.png')


def add_node_to_graph(graph, node, data_holder):
    if node.value.arity == 0:
        return
    graph.add_edge(node, node.rightChild, label='no keyword')
    right_edge = graph.get_edge(node, node.rightChild)
    right_edge.attr['color'] = RED_COLOR
    graph.add_edge(node, node.leftChild, label='has keyword')
    left_edge = graph.get_edge(node, node.leftChild)
    left_edge.attr['color'] = GREEN_COLOR
    style_node(node, graph, data_holder)
    style_node(node.leftChild, graph, data_holder)
    style_node(node.rightChild, graph, data_holder)

    graph.get_node(node.rightChild).attr["label"] = node.rightChild.value.name
    graph.get_node(node.leftChild).attr["label"] = node.leftChild.value.name
    add_node_to_graph(graph, node.leftChild, data_holder)
    add_node_to_graph(graph, node.rightChild, data_holder)


def style_node(node, graph, data_holder):
    if BASE_FEATURE_NAME in node.value.name:
        graph_node = graph.get_node(node)
        graph_node.attr["label"] = get_primitive_keyword(node.value.name, data_holder.features)
        graph_node.attr["shape"] = "box"
    else:
        graph_node = graph.get_node(node)
        graph_node.attr["label"] = node.value.name
        graph_node.attr["shape"] = "egg"
        graph_node.attr["style"] = "filled"
        if node.value.name == 'Count':
            graph_node.attr["fillcolor"] = GREEN_COLOR
            graph_node.attr["color"] = GREEN_COLOR
        else:
            graph_node.attr["fillcolor"] = RED_COLOR
            graph_node.attr["color"] = RED_COLOR


def get_primitive_keyword(name, features):
    return features[int(name.replace(BASE_FEATURE_NAME, ''))]

