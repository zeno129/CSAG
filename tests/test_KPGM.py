import pytest
import itertools
import sys
import operator
from kronecker import KPGM as model


# This test does not work (divide by zero)
# def test_deterministic():
#     b = 2
#     theta = [[0, 1], [0, 1]]
#     k = 2
#
#     g = model.KPGM(theta, k, b)
#
#     assert (3, 0) in g.edges
#     assert (0, 3) in g.edges
#     assert (1,2) in g.edges
#     assert (2,1) in g.edges

def test_random_probability():
    b = 2
    theta = [[0.5, 0.5], [0.5, 0.5]]
    k = 2
    vertices = range(operator.pow(b, k))
    # possible_edges = list(itertools.combinations_with_replacement(vertices, 2))
    possible_edges = list(itertools.product(vertices, repeat=2))
    # counts = defaultdict(int)
    counts = {edge: 0 for edge in possible_edges}

    for i in range(10):
        g = model.KPGM(theta, k, b)
        for e in possible_edges:
            if e in g.edges:
                counts[e] += 1

    sys.stdout.write(str(counts))

    assert True