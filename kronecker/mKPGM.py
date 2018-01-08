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
from KPGM import KPGM


class mKPGM:
    def __init__(self, theta, K, b, l):
        # (2) Generate G_l with KPGM - - - - *
        G_l = KPGM(theta, l, b)
        self.igraph = G_l.igraph
        self.edges = G_l.edges  # list of tuples

        # (1) Add vertices - - - - - - - - - *
        self.vertices = operator.pow(b, K)
        self.igraph.add_vertices(self.vertices)

        # (3) Construct U; Unique probability values
        # theta[i][j] does this

        for k in range(l, K):  # (4) iterate
            # (5) Edges for this iteration
            E_k = []

            # (6) Indices of edges in G_{k-1}, combined Lambda_q and Lambda_r
            Lambda_ij = self.edges

            for i in range(b):  # (7) iterate
                for j in range(b):  # (8) iterate
                    # (9) pi' k-th unique probability
                    theta_ij = theta[i][j]

                    # (10) num. of edges previous layer
                    T_ij = len(self.edges)

                    # (11) Replace binomial sampling with
                    # Geometric group probability sampling (GGPS), line 6
                    countEdge = 1 + floor(np.log(1 - np.random.uniform(0, 1)) / np.log(1 - theta_ij))

                    # (13) loop replaced by GGPS line 7
                    while countEdge <= T_ij:
                        # (12) Random permutation
                        sigma = range(K)
                        random.shuffle(sigma)

                        # (14) Generate new edge (u,v); GGPS line 8
                        # i.e., pick a random edge from G_{l+k-1} and calculate new indices
                        (u, v) = random.sample(self.edges, 1)[0]
                        u = u * b + i
                        v = v * b + j

                        # (15) Add edge to list; GGPS line 9
                        E_k.append((u,v))

                        # self.igraph.add_edge(u,v); GGPS line 10
                        countEdge += 1 + floor(np.log(1 - np.random.uniform(0, 1)) / np.log(1 - theta_ij))

            # Add edges to igraph
            self.igraph.add_edges(E_k)
            self.edges.extend(E_k)

    def write_igraph(self, filepath):
        self.igraph.write_pickle(filepath)

if __name__ == '__main__':
    my_b = 2
    my_K = 10
    my_l = 5
    my_theta = [[0.99, 0.55], [0.55, 0.75]]
    graph = mKPGM(my_theta, my_K, my_b, my_l)

    out_dir = '/Users/giselle/Development/research/gen_data'
    filename = '20171203_graph-b_%s-K_%s.pkl' % (my_b, my_K)

    d = os.path.dirname(out_dir)
    if not os.path.exists(d):
        os.makedirs(d)

    filepath = os.path.join(out_dir, filename)

    graph.write_igraph(filepath)
