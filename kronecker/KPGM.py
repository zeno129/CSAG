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


class KPGM:
    def __init__(self, theta, K, b):
        # Create igraph
        self.igraph = igraph.Graph()

        # Add vertices
        self.vertices = operator.pow(b, K)
        self.igraph.add_vertices(self.vertices)

        edges = []

        all_theta_indices = [(i, j) for i in range(b) for j in range(b)]

        combinations = itertools.combinations_with_replacement(all_theta_indices, K)
        # Construct U; Unique probability values
        U = [c for c in combinations]  # theta indices to create all pi'_k

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

        # Add edges to igraph
        self.igraph.add_edges(edges)
        self.edges = edges

    def write_igraph(self, filepath):
        self.igraph.write_pickle(filepath)

if __name__ == '__main__':
    my_b = 2
    # my_K = 10
    my_K = 2
    # my_theta = [[0.99, 0.55], [0.55, 0.75]]
    my_theta = [[0, 1], [0, 1]]
    graph = KPGM(my_theta, my_K, my_b)

    print(graph.edges)

    # out_dir = '/Users/giselle/Development/research/gen_data'
    # filename = '20171203_graph-b_%s-K_%s.pkl' % (my_b, my_K)
    #
    # d = os.path.dirname(out_dir)
    # if not os.path.exists(d):
    #     os.makedirs(d)
    #
    # filepath = os.path.join(out_dir, filename)
    #
    # graph.write_igraph(filepath)

