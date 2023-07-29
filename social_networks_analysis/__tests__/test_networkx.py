import pytest
import networkx as nx
import numpy as np


def test_networkx1():
    print("test_networkx1")
    graph = nx.Graph()
    graph.add_nodes_from([10, 11, 12, 13, 14, 15])
    graph.add_edges_from([(10, 11), (14, 15), (11, 10), (12, 15)])
    graph_array = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
    assert (len(graph.nodes) == 6)
    assert (len(graph.edges) == 3)

    graph2 = nx.Graph()
    graph2.add_nodes_from([15, 14, 13, 12, 11, 10])
    graph2.add_edges_from([(14, 15), (10, 11), (12, 15), (11, 10)])
    graph2_array = nx.to_numpy_array(graph2, nodelist=sorted(graph2.nodes()))
    assert (len(graph2.nodes) == 6)
    assert (len(graph2.edges) == 3)
    assert (graph_array == graph2_array).all()


def test_networkx_shortest_path_length():
    print("test_networkx_shortest_path_length")
    # n * (n-1) / 2 + n
    graph = nx.Graph()
    # graph.add_nodes_from([10, 11, 12, 13, 14, 15])
    graph.add_nodes_from([15, 14, 13, 12, 11, 10, 9])
    graph.add_edges_from([(10, 11), (14, 15), (11, 10), (11, 12), (12, 15)])
    graph_array = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))

    dgeodesic = nx.shortest_path_length(graph)
    dgeodesic_list = list(dgeodesic)
    print('dgeodesic_list', dgeodesic_list)
