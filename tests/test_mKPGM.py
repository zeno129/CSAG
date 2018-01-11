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
