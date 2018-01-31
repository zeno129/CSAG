# import pytest
import itertools
import operator
import numpy as np
from kronecker import mKPGM as model


# def test_equal_probabilities_edges():
#     b = 2
#     theta = [[0.5, 0.5], [0.5, 0.5]]
#     k = 5
#     l = 2
#     vertices = range(operator.pow(b, k))
#     n = 100
#
#     possible_edges = list(itertools.product(vertices, repeat=2))
#     counts = {edge: 0 for edge in possible_edges}
#
#     for i in range(n):
#         g = model.mKPGM(theta, k, b, l)
#         for e in possible_edges:
#             if e in g.edges:
#                 counts[e] += 1
#
#     p = np.power(theta[0][0], k)
#     mean = n * p
#     variance = n * p * (1 - p)
#
#     # for cnt in counts.values():
#     #     assert float(cnt)/n >= mean - variance
#     #     assert float(cnt)/n <= mean + variance
#
#     cnt_avg = np.mean(counts.values())
#     assert cnt_avg >= mean - variance
#     assert cnt_avg <= mean + variance
#
#
# def test_diff_probabilities_edges():
#     b = 2
#     k = 5
#     l = 2
#     theta = [[0.7, 0.4], [0.4, 0.5]]
#     vertices = range(operator.pow(b, k))
#     n = 100
#
#     possible_edges = list(itertools.product(vertices, repeat=2))
#     counts = {edge: 0 for edge in possible_edges}
#
#     for i in range(n):
#         g = model.mKPGM(theta, k, b, l)
#         for e in possible_edges:
#             if e in g.edges:
#                 counts[e] += 1
#
#     P_k = np.kron(np.array(theta), np.array(theta))
#     for t in range(k-1):
#         P_k = np.kron(np.array(P_k), np.array(theta))
#
#     for (i,j) in counts.keys():
#         p = P_k[i][j]
#         mean = n * p
#         variance = n * p * (1 - p)
#
#         cnt = counts[(i,j)]
#
#         assert cnt >= mean - variance
#         assert cnt <= mean + variance

def test_different_probabilities_avg_edges():
    b = 2
    k = 5
    l = 2
    theta = [[0.7, 0.4], [0.4, 0.5]]
    vertices = range(operator.pow(b, k))
    n = 100

    possible_edges = list(itertools.product(vertices, repeat=2))
    counts = {edge: 0 for edge in possible_edges}

    for i in range(n):
        g = model.mKPGM(theta, k, b, l)
        for e in possible_edges:
            if e in g.edges:
                counts[e] += 1

    S = sum(theta[0]) + sum(theta[1])
    S_2 = np.square(theta[0][0]) + np.square(theta[0][1]) \
        + np.square(theta[1][0]) + np.square(theta[1][1])

    exp_num_edges = np.power(S, k)
    var_num_edges = np.power(S, k-1) * (np.power(S, k-l) - 1) \
                * float(S-S_2)/(S-1) + (np.power(S, k-l) - np.power(S_2, l)) \
                * np.power(S, 2*(k-l))

    avg_edges = float(sum(counts.values()))/n

    assert avg_edges >= exp_num_edges - var_num_edges
    assert avg_edges <= exp_num_edges + var_num_edges
