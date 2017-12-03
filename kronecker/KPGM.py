from __future__ import division
import igraph
import scipy.io as sio
import os, sys
import pickle
import itertools
from math import factorial, floor
import operator
import functools
import numpy as np
import random


def group_sampling(theta, K, b):
    num_vertices = operator.pow(b, K)
    edges = []

    g = igraph.Graph()  # Create graph
    g.add_vertices(num_vertices)  # Add nodes

    all_theta_indices = [(i, j) for i in range(b) for j in range(b)]

    combinations = itertools.combinations_with_replacement(all_theta_indices, K)
    U = [c for c in combinations]

    for k in range(len(U)):
        # Obtain pi' k-th unique probability
        Lambda_ij = U[k]
        thetas = [theta[i][j] for (i,j) in Lambda_ij]
        pi_prime_k = (functools.reduce(lambda x, y: x*y, thetas))

        Gamma_k = [U[k].count(idx) for idx in all_theta_indices]

        # Calculate T_k
        gamma_factorials = [factorial(gamma) for gamma in Gamma_k]
        # Ref: https://docs.python.org/2/library/functions.html#reduce
        T_k = factorial(K) / (functools.reduce(lambda x, y: x*y, gamma_factorials))

        countEdge = 1 + floor(np.log(1 - np.random.uniform(0, 1)) / np.log(1 - pi_prime_k))

        while countEdge <= T_k:
            # Random permutation
            sigma = range(K)
            random.shuffle(sigma)

            # Generate new edge (u,v)
            # Note: range starts at 0, so we don't need t do l-1
            # Python indices start at 0, not 1
            u = sum([Lambda_ij[sigma[l]][0] * operator.pow(b, l) + 1 for l in range(K)])
            v = sum([Lambda_ij[sigma[l]][1] * operator.pow(b, l) + 1 for l in range(K)])

            # Add edge to list
            # iGraph starts vertices at 0
            # For some reason, resulting edges start at index 9, not 0?
            edges.append((u-K, v-K))

            # print("u=%s\tv=%s" % (u-1, v-1))
            # g.add_edge(u-1, v-1)

            countEdge += 1 + floor(np.log(1 - np.random.uniform(0, 1)) / np.log(1 - pi_prime_k))

    # Add edges to graph
    g.add_edges(edges)

    return g

if __name__ == '__main__':
    my_b = 2
    my_K = 10
    my_theta = [[0.99, 0.55], [0.55, 0.75]]
    graph = group_sampling(my_theta, my_K, my_b)

    out_dir = '/Users/giselle/Development/research/gen_data'
    fname = '20171203_graph-b_%s-K_%s.pkl' % (my_b, my_K)
    # filepath = os.path.join(directory, fname)

    d = os.path.dirname(out_dir)
    if not os.path.exists(d):
        os.makedirs(d)

    filepath = os.path.join(out_dir, fname)
    graph.write_pickle(filepath)
