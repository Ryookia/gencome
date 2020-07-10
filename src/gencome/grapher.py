import pygraphviz as pgv

import gencome.config
from gencome.utils import get_primitive_keyword

RED_COLOR = "#ff6666"
GREEN_COLOR = '#66ff99'


def visualize_individual(head_node):
    graph = pgv.AGraph(strict=False, directed=True)
    add_node_to_graph(graph, head_node)
    graph.layout(prog='dot')
    return graph


def add_node_to_graph(graph, node):
    if node.value.arity == 0:
        return
    graph.add_edge(node, node.leftChild, label='true', fontname="Arial", fontcolor=GREEN_COLOR)
    left_edge = graph.get_edge(node, node.leftChild)
    left_edge.attr['color'] = GREEN_COLOR
    graph.add_edge(node, node.rightChild, label='false', fontname="Arial", fontcolor=RED_COLOR)
    right_edge = graph.get_edge(node, node.rightChild)
    right_edge.attr['color'] = RED_COLOR
    style_node(node, graph)
    style_node(node.leftChild, graph)
    style_node(node.rightChild, graph)

    graph.get_node(node.rightChild).attr["label"] = f"{node.rightChild.value.name}"
    graph.get_node(node.leftChild).attr["label"] = f"{node.leftChild.value.name}"
    add_node_to_graph(graph, node.leftChild)
    add_node_to_graph(graph, node.rightChild)


def style_node(node, graph):
    if gencome.config.BASE_FEATURE_NAME in node.value.name:
        graph_node = graph.get_node(node)
        graph_node.attr["label"] = f"'{get_primitive_keyword(node.value.name, gencome.config.features)}'"
        graph_node.attr["shape"] = "box"
        graph_node.attr["fontname"] = "Arial"
    else:
        graph_node = graph.get_node(node)
        graph_node.attr["label"] = f"{node.value.name}"
        graph_node.attr["shape"] = "egg"
        graph_node.attr["style"] = "filled"
        graph_node.attr["fontname"] = "Arial"
        if node.value.name == gencome.config.COUNT_LABEL:
            graph_node.attr["fillcolor"] = GREEN_COLOR
            graph_node.attr["color"] = GREEN_COLOR
        else:
            graph_node.attr["fillcolor"] = RED_COLOR
            graph_node.attr["color"] = RED_COLOR