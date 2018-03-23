from __future__ import division
import igraph
from math import floor
import operator
import numpy as np
import random
from KPGM import KPGM


class mKPGM:
    def __init__(self, theta, K, b, l):
        self.theta = theta
        self.K = K
        self.b = b
        self.l = l

        # Empty igraph
        self.igraph = None

        # (2) Generate G_l with KPGM - - - - *
        G_l = KPGM(theta, l, b)

        self.edges = G_l.edges  # list of tuples
        # TODO: get blocks from KPGM???
        self.blocks = []
        # self.blocks = G_l.get_block(l)

        # (1) Add vertices - - - - - - - - - *
        self.vertices = operator.pow(b, K)

        # (3) Construct U; Unique probability values
        # theta[i][j] does this

        for k in range(l, K):  # (4) iterate
            # (5) Edges for this iteration
            E_k = []

            # (6) Indices of edges in G_{k-1}, combined Lambda_q and Lambda_r
            Lambda_ij = self.edges
            self.blocks.append(dict.fromkeys([item for row in theta for item in row], list()))
            # self.blocks.append({'prob': None, 'edges': []})

            for i in range(b):  # (7) iterate
                for j in range(b):  # (8) iterate
                    # (9) pi' k-th unique probability
                    theta_ij = theta[i][j]
                    # Blocks: store pi'_k with k
                    # self.blocks.append({'prob': theta_ij, 'edges': []})
                    # self.blocks[k][theta_ij]

                    # (10) num. of edges previous layer
                    T_ij = len(self.edges)

                    # (11) Replace binomial sampling with
                    # Geometric group probability sampling (GGPS), line 6
                    countEdge = 1 + floor(np.log(1 - np.random.uniform(0, 1)) / np.log(1 - theta_ij))

                    # (13) loop replaced by GGPS line 7
                    while countEdge <= T_ij:
                        # (12) Random permutation
                        # sigma = range(K)
                        # random.shuffle(sigma)

                        # (14) Generate new edge (u,v); GGPS line 8
                        # pick a random edge from G_{l+k-1}
                        # (u, v) = random.sample(Lambda_ij, 1)[0]
                        (u, v) = random.sample(self.edges, 1)[0]

                        # Blocks: store u,v with k
                        # self.blocks[k]['edges'].append((u, v))
                        self.blocks[k - l][theta_ij].append((u, v))

                        # calculate new indices
                        u = u * b + i
                        v = v * b + j

                        # (15) Add edge to list; GGPS line 9
                        E_k.append((u,v))

                        # GGPS line 10
                        countEdge += 1 + floor(np.log(1 - np.random.uniform(0, 1)) / np.log(1 - theta_ij))

            # Add edges to igraph
            self.edges.extend(E_k)

    def create_igraph(self):
        # Empty igraph
        self.igraph = igraph.Graph()
        # self.igraph = self.G_l.igraph.create_igraph()

        # Add vertices
        self.igraph.add_vertices(self.vertices)

        # Add edges to igraph
        self.igraph.add_edges(self.edges)

    def write_igraph(self, filepath):
        if not self.igraph:
            self.create_igraph()

        self.igraph.write_pickle(filepath)

    def get_block(self, num):
        # TODO: Retrieve probability blocks
        return self.blocks[num - self.l - 1]
