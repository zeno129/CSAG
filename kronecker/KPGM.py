from __future__ import division
import igraph
import itertools
from math import factorial, floor
import operator
import functools
import numpy as np
import random


class KPGM:
    def __init__(self, theta, K, b):
        self.b = b
        self.K = K
        self.theta = theta

        # Empty igraph
        self.igraph = None

        # Add vertices
        self.vertices = operator.pow(b, K)

        # Init list of edges
        edges = []

        # TODO Init list of "blocks"
        # self.blocks = []

        all_theta_indices = [(i, j) for i in range(b) for j in range(b)]

        combinations = itertools.combinations_with_replacement(all_theta_indices, K)
        # Construct U; Unique probability values
        U = [c for c in combinations]  # theta indices to create all pi'_k

        for k in range(len(U)):
            # Obtain pi' k-th unique probability
            Lambda_ij = U[k]

            thetas = [theta[i][j] for (i,j) in Lambda_ij]
            pi_prime_k = (functools.reduce(lambda x, y: x*y, thetas))
            # TODO Blocks: store pi'_k with k
            # self.blocks.append({'prob': pi_prime_k, 'edges': []})
            # self.blocks.append({'prob': pi_prime_k, 'edges': []})

            # Num. of times theta_ij occurs in calculation of pi_prime_k
            Gamma_k = [U[k].count(idx) for idx in all_theta_indices]

            # Calculate T_k -- all possible permutations of Gamma_k
            gamma_factorials = [factorial(gamma) for gamma in Gamma_k]
            # Ref: https://docs.python.org/2/library/functions.html#reduce
            T_k = factorial(K) / (functools.reduce(lambda x, y: x*y, gamma_factorials))

            countEdge = 1 + floor(np.log(1 - np.random.uniform(0, 1)) / np.log(1 - pi_prime_k))

            while countEdge <= T_k:
                # Random permutation
                sigma = range(K)
                random.shuffle(sigma)

                # Generate new edge (u,v)
                # Note: range starts at 0, so we don't need t do l-1, or + 1
                # Python indices start at 0, not 1
                u = sum([Lambda_ij[sigma[l]][0] * operator.pow(b, l) for l in range(K)])
                v = sum([Lambda_ij[sigma[l]][1] * operator.pow(b, l) for l in range(K)])

                # Add edge to list
                # iGraph starts vertices at 0
                edges.append((u, v))
                # TODO Blocks: store u,v with k
                # self.blocks[k]['edges'].append((u-K, v-K))

                countEdge += 1 + floor(np.log(1 - np.random.uniform(0, 1)) / np.log(1 - pi_prime_k))

        self.edges = edges

    def create_igraph(self):
        # Empty igraph
        self.igraph = igraph.Graph()

        # Add vertices
        self.igraph.add_vertices(self.vertices)

        # Add edges to igraph
        self.igraph.add_edges(self.edges)

    def write_igraph(self, filepath):
        if not self.igraph:
            self.create_igraph()

        self.igraph.write_pickle(filepath)

    def get_block(self, num):
        '''

        :param num: number of Kronecker multiplications
        :return:
        '''

        # block = dict()
        #
        # # TODO: (1) Calculate unique probabilities - - - - - - - - - - - - - *
        # all_theta_indices = [(i, j) for i in range(self.b) for j in range(self.b)]
        # combinations = itertools.combinations_with_replacement(all_theta_indices, num)
        # # Construct U; Unique probability values
        # U = [c for c in combinations]  # theta indices to create all pi'_k
        #
        # for k in range(len(U)):  # for each unique prob.
        #     Lambda_ij = U[k]  # theta i,j indices for pi'_k
        #     theta_values = [self.theta[i][j] for (i, j) in Lambda_ij]
        #     # pi' k-th unique probability
        #     pi_prime_k = (functools.reduce(lambda x, y: x * y, theta_values))
        #     block[pi_prime_k] = list()
        #
        #     # sigma = range(num)
        #
        #     # Num. of times theta_ij occurs in calculation of pi_prime_k
        #     Gamma_k = [U[k].count(idx) for idx in all_theta_indices]
        #     # Calculate T_k -- all possible permutations of Gamma_k
        #     gamma_factorials = [factorial(gamma) for gamma in Gamma_k]
        #     T_k = factorial(K) / (functools.reduce(lambda x, y: x * y, gamma_factorials))
        #
        #
        # # TODO: (2) Calculate edge locations
        # # TODO: Retrieve probability blocks
        # # return self.blocks[num - 1]
        # pass
        raise NotImplementedError
