import pytest
import itertools
import sys
import operator
import numpy as np
from kronecker import KPGM as model


def test_equal_probabilities_edges():
    b = 2
    theta = [[0.5, 0.5], [0.5, 0.5]]
    k = 2
    vertices = range(operator.pow(b, k))
    n = 100

    possible_edges = list(itertools.product(vertices, repeat=2))
    counts = {edge: 0 for edge in possible_edges}

    for i in range(n):
        g = model.KPGM(theta, k, b)
        for e in possible_edges:
            if e in g.edges:
                counts[e] += 1

    p = 0.25
    mean = n * p
    variance = n * p * (1 - p)

    for cnt in counts.values():
        assert cnt >= mean - variance
        assert cnt <= mean + variance


def test_diff_probabilities_edges():
    b = 2
    k = 2
    theta = [[0.7, 0.4], [0.4, 0.5]]
    vertices = range(operator.pow(b, k))
    n = 100

    possible_edges = list(itertools.product(vertices, repeat=2))
    counts = {edge: 0 for edge in possible_edges}

    for i in range(n):
        g = model.KPGM(theta, k, b)
        for e in possible_edges:
            if e in g.edges:
                counts[e] += 1

    P_k = np.kron(np.array(theta), np.array(theta))

    for (i,j) in counts.keys():
        p = P_k[i][j]
        mean = n * p
        variance = n * p * (1 - p)

        cnt = counts[(i,j)]

        assert cnt >= mean - variance
        assert cnt <= mean + variance


# TODO: Figure out why KPGM not producing random graphs
def test_random_graph_clustering():
    b = 2
    prob = 0.65
    theta = [[prob, prob], [prob, prob]]
    k = 5
    vertices = range(operator.pow(b, k))
    n = 100

    possible_edges = list(itertools.product(vertices, repeat=2))
    counts = {edge: 0 for edge in possible_edges}

    p = np.power(prob, k)

    clusterings = []

    for i in range(n):
        g = model.KPGM(theta, k, b)
        g.create_igraph()
        graph = g.igraph
        graph.to_undirected
        # graph = g.igraph.clusters().giant()

        # clustering_coeffs = []
        #
        # for node in graph.vs():
        #     neighbors = graph.vs.select(graph.neighborhood(node))
        #     num_neighbors = len(neighbors)
        #     possible_edges = num_neighbors * num_neighbors
        #     edges = 0
        #
        #     for n in neighbors:
        #         neighbors2 = graph.vs.select(graph.neighborhood(n))
        #         edges += len([n2 for n2 in neighbors2 if n2 in neighbors])
        #
        #     clustering_coeffs.append(float(edges) / possible_edges)
        #
        # clusterings.append(np.mean(clustering_coeffs))


        # clustering_coeff = np.round(graph.transitivity_avglocal_undirected(), decimals=2)
        clustering_coeff = np.round(graph.transitivity_avglocal_undirected(mode="zero"), decimals=2)
        clusterings.append(clustering_coeff)


    mean = float(np.round(np.mean(clusterings), decimals=2))

    assert p <= mean + 0.1 and p >= mean - 0.1

def test_random_graph_density():
    b = 2
    prob = 0.65
    theta = [[prob, prob], [prob, prob]]
    k = 5
    vertices = range(operator.pow(b, k))
    n = 100

    possible_edges = list(itertools.product(vertices, repeat=2))
    counts = {edge: 0 for edge in possible_edges}

    p = np.round(np.power(prob, k), decimals=2)

    densities = []

    for i in range(n):
        g = model.KPGM(theta, k, b)
        g.create_igraph()
        graph = g.igraph

        N = graph.vcount()
        E = graph.ecount()
        # D = (2 * E) / float(N * (N - 1))
        D = (E) / float(N * (N - 1))

        densities.append(D)

    mean = np.round(np.mean(densities), decimals=2)

    assert p <= mean + 0.025 and p >= mean - 0.025
