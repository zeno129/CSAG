import pytest
import itertools
import sys
import operator
from kronecker import KPGM as model


def test_equal_probabilities():
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

    # sys.stdout.write(str(counts))

    p = 0.25
    mean = n * p
    std_dev = n * p * (1 - p)

    for cnt in counts.values():
        assert cnt >= mean - std_dev
        assert cnt <= mean + std_dev
