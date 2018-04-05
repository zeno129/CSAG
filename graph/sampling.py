import random
import numpy as np
import itertools
from scipy.stats.stats import pearsonr
from scipy.optimize import linprog
from kronecker import mKPGM as mKPGM


def graph_sampling(graphIn, xIn, model, epsilon, distribution, params_test=None):
    """
    Graph Sampling algorithm

    :param graphIn: contains list of vertices and list of edges
    :type graphIn: tuple

    :param xIn: node attributes for graphIn
    :type xIn: list

    :param model: GNM and parameters
    :type model: dict

    :param epsilon: error
    :type epsilon: float

    :param distribution: "binomial" or "multinomial:
    :type distribution: String
    
    :return: graphOut (graph), xOut (attributes), rhoOut (correlation)
    :rtype: tuple, list, float
    """

    verticesIn, edgesIn = graphIn

    # model = ??? -- is this just b and l? ...since theta and K can be learned
    # thetaG=None, beta=None
    # TODO (1) learn parameters

    # TODO: testing code
    psi = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # psi = list(itertools.product(thetaX.keys(), repeat=2))

    if params_test:
        # if "beta" in params_test.keys():
        beta = params_test["beta"]

        # if "thetaG" in params_test.keys():
        thetaG = params_test["thetaG"]
        thetaX = f_x(xIn, distribution)

    else:
        psi, beta, thetaX, thetaG = learn_parameters(graphIn, xIn, model, distribution)

    if not params_test:
        psi, beta, thetaX, thetaG = learn_parameters(graphIn, xIn, model, distribution)
        n = pow(model['b'], model['K'])
        # (2) sample node attributes xOut from P(X|thetaX)
        xOut = sample_x(thetaX, distribution, n)
    else:
        # if "beta" in params_test.keys():
        beta = params_test["beta"]

        # if "thetaG" in params_test.keys():
        thetaG = params_test["thetaG"]
        thetaX = f_x(xIn, distribution)
        # TODO: add param to hardcode or sample
        xOut = xIn

    # thetaG = [[0.7, 0.4], [0.4, 0.5]]
    # graphOut = model.mKPGM(thetaG, K=5, b=2, l=2)

    # (3) init rhoOut = (correlation) and l_o = K - l - 1
    rhoOut = np.inf
    if model['name'] == "mKPGM":
        l_o = model['K'] - model['l'] - 1
    else:
        # TODO: implement for KPGM
        raise NotImplemented
    # K can be learned
    # l has to be specified
    # rho_OUT = np.inf
    # l_o = K - l - 1
    # TODO: (4) while loop --
    # TODO: (5-8) block sampling with ME and LP

    # (5) sample last block
    # TODO: implement for KPGM
    # g = mKPGM.mKPGM(thetaG, model['K'], model['b'], model['l'])
    graphOut = mKPGM.mKPGM(thetaG, model['K'], len(thetaG[0]), model['l'])
    # (8) sample edges

    # TODO: testing version
    if params_test and "last_block" in params_test and params_test["last_block"]:
        verticesOut, edgesOut = maxent_edge_sampling(model, thetaG, params_test["last_block"], psi, beta, xOut)
    else:
        verticesOut, edgesOut = maxent_edge_sampling(model, thetaG, graphOut.blocks[-1], psi, beta, xOut)

    # verticesOut, edgesOut = maxent_edge_sampling(model, thetaG, graphOut.blocks[-1], psi, beta, xOut)
    graphOut.edges = edgesOut

    # TODO: (9) calculate rhoOut
    # Initial version
    # TODO: use graphOut.edges
    rhoOut = calc_correlation(graphOut.edges, xOut)
    # rhoOut = calc_correlation(edgesIn, xOut)

    # TODO: (10) update l_o

    return graphOut, xOut
    # return graphIn, xOut


def learn_parameters(graphIn, xIn, model, distribution):
    """

    :param graphIn: contains list of vertices and edges
    :type graphIn: tuple

    :param xIn: node attributes for graphIn
    :type xIn: list

    :param model: GNM and parameters
    :type model: dict

    :param distribution: "binomial" or "multinomial"
    :type distribution: String

    :return: psi (edge types), beta (fraction of edge types),
    thetaX (parameters for P(X)), thetaG (parameters for P(G))
    :rtype: list of tuples, dictionary, dictionary, matrix
    """
    # TODO: (1) learn parameters (psi, beta, thetaX, thetaG)
    # TODO: do I need model in here??
    psi = None
    beta = None
    thetaX = f_x(xIn, distribution)
    thetaG = [[0.7, 0.4], [0.4, 0.5]]

    return psi, beta, thetaX, thetaG


def f_x(xIn, distribution):
    """
    function to learn thetaX parameters for P(X)

    :param xIn: attributes for vertices of graphIn
    :type xIn: list

    :param distribution: "binomial" or "multinomial:
    :type distribution: String

    :return:
    """
    # (1) learn parameters thetaX

    if distribution in ["binomial", "multinomial"]:
        labels = list(set(xIn))
        thetaX = {}

        for l in labels:
            thetaX[l] = float(xIn.count(l)) / len(xIn)

        return thetaX
    else:
        raise ValueError("Supported distributions are 'binomial' and 'multinomial'")


def sample_x(thetaX, distribution, num_samples):
    """
    Sample node attributes xOut from P(X|thetaX)

    :param thetaX: parameters for P(X)
    :type thetaX: dictionary

    :param distribution: "binomial" or "multinomial:
    :type distribution: String

    :param num_samples: number of samples
    :type num_samples: int

    :return: xOut (new attributes for graphOut)
    :rtype: list
    """

    if distribution in ["binomial", "multinomial"]:
        labels = thetaX.keys()
        probabilities = [thetaX[l] for l in labels]

        # tmp = np.random.multinomial(n=(len(labels) - 1), pvals=probabilities, size=num_samples)
        tmp = np.random.multinomial(n=1, pvals=probabilities, size=num_samples)
        xOut = [t[0] for t in tmp]

        return xOut
    else:
        raise ValueError("Supported distributions are 'binomial' and 'multinomial'")


def maxent_edge_sampling(model, thetaG, block, l, psi, beta, xOut):
    """

    :param model: GNM and parameters
    :type model: dict

    :param thetaG: parameters for marginal distribution of network structure P(G)
    :type thetaG: matrix

    :param block: sample block from penultimate iteration of mKPGM
    :type block: matrix

    :param psi: edge types
    :type psi: list

    :param beta: fraction of edges of each type
    :type beta: list

    :param xOut: node attributes
    :type xOut: list

    :return: graphOut (output graph, contains num. of vertices and edge list)
    :rtype: tuple
    """
    # U = unique probabilities
    # T = edge locations
    # (3)
    U, T = get_unique_prob_edge_location(model, thetaG, block, psi, xOut)
    # N_e = 0
    Nus = []

    # for each unique prob. pi_u
    for pi_u in U:
        # (5) Draw num. edges to sample per unique prob.
        n_u = np.random.binomial(len(T[pi_u]), pi_u)
        Nus.append(n_u)  # (6)
        # Accum. total num. edges to be sampled
        # N_e += n_u
    N_e = sum(Nus)

    # (7) Draw num. edges per edge-type to match rho_IN
    gamma = list(np.random.multinomial(n=N_e, pvals=beta, size=1))
    # n = Number of experiments (int)
    # pvals = Probabilities of each of the p different outcomes (sequence of floats, length p)
    # size = Output shape (int or tuple of ints, optional)

    E_OUT = []
    for i, u in enumerate(U):
        # (9) Draw num. edges per edge type for pi_u
        Y = list(np.random.multinomial(n=Nus[i], pvals=[np.double(g)/N_e for g in gamma[0]], size=1))

        # (10 - 14)
        for j, p in enumerate(psi):
            # (11) Sampling Y_j edges at random from T_uj possible locations
            possible_edges = list(T[u][p])
            random.shuffle(possible_edges)
            edges = possible_edges[:Y[0][j]]

            E_OUT.extend(edges)  # (12)
            gamma[0][j] -= Y[0][j]  # (13)
            N_e -= Y[0][j]  # (14)

    # vertices = len(block[0])
    vertices = pow(model['b'], model['K'])

    return vertices, E_OUT


def get_unique_prob_edge_location(model, thetaG, block, psi, xOut):
    """

    :param model: GNM and parameters
    :type model: dict

    :param thetaG: parameters for marginal distribution of network structure P(G)
    :type thetaG: matrix

    :param block: sample block from penultimate iteration of mKPGM
    :type block: dict

    :param psi: edge types
    :type psi: list

    :param xOut: node attributes
    :type xOut: list

    :return: U (unique probabilities), T (edge locations)
    :rtype: set, matrix
    """

    if model['name'] == "mKPGM":
        # Calc U (set unique probabilities), use node attributes
        # For mKPGM it's just the theta[i][j] values
        # U = thetaG.flatten()
        U = [i for row in thetaG for i in row]

        # Index T by probability (pi_u) and edge-type (psi)
        T = dict.fromkeys(U, dict.fromkeys(psi, list()))

        b = model['b']

        # get indices for edges (non-zero probability)
        for prob in block.keys():  # these correspond to theta values for mKPGM
            for s,t in block[prob]:
                # map i,j from block[l] to u,v in E_OUT
                for i in range(b):
                    for j in range(b):
                        u = s * b + i
                        v = t * b + j

                        edge_type = (xOut[u], xOut[v])
                        edge_loc = (u, v)
                        T[prob][edge_type].append(edge_loc)

        return U, T
    else:
        # TODO: calc U and T for KPGM
        raise NotImplemented


def lp_block_search(model, thetaG, blockSample_l, l, psi, beta, xOut):
    """

    :param model: GNM and parameters
    :type model: dict

    :param thetaG: parameters for marginal distribution of network structure P(G)
    :type thetaG: matrix

    :param blockSample_l: sample block from l-th iteration of mKPGM
    :type block: matrix

    :param psi: edge types
    :type psi: list

    :param beta: fraction of edges of each type
    :type beta: list

    :param xOut: node attributes
    :type xOut: list

    :return:
        :blockSample_lPlus1: sampled block in l+1
    """

    # (3)
    U, T = get_unique_prob_block_location(model, thetaG, blockSample_l, l, psi, xOut)

    # for each unique prob. pi_u
    # (4)
    for u, pi_u in enumerate(U):
        # Draw num. blocks to sample per unique prob.
        T_u = [item for item in T[pi_u]]
        n_u = np.random.binomial(len(T_u), pi_u)  # (5)
        # n_u = np.random.binomial(len(T[pi_u]), pi_u)

        e = []
        for j, psi_j in enumerate(psi):  # (6)
            # (7) fraction of possible edges leading to rho_IN
            e.append(beta[j] * n_u)

        # (8) Determing A_jk --
        # Num. descendant edges of type psi_j in psi
        # per position t_k in T_u

        # TODO: change from T_u to N_omega
        # rows are for each edge-type psi_j
        # cols are for each location t_k in T_u
        A = [[len(t_k[psi_j]) for t_k in T[pi_u]] for psi_j in psi]

        # A = []
        # for psi_j in psi:
        #     A_j = []
        #     for t_k in T[pi_u]:
        #         A_jk = len(t_k[psi_j])
        #         A_j.append(A_jk)
        #     A.append(A_j)

        ub = []
        for k, t_k in enumerate(T[pi_u]):  # (9)
            ub_k = sum([A[j][k] for j in range(len(psi))])
            ub.append(ub_k)

        # (11) TODO: (REVISE) solve linear equation
        c = np.array([[-1 * item for item in row] for row in A])
        Aeq = np.ones((1, len(T[pi_u])))
        beq = np.array([n_u])
        bounds = np.array([(0, ub_k) for ub_k in ub])
        linprog(c=c, A_ub=np.array(A), b_ub=np.array(e),
                A_eq=Aeq, b_eq=beq, bounds=bounds,
                method='interior-point')


        # (12) TODO: sample block
        for k in range(len(T[pi_u])):
            # (13) Sampling X_j blocks at random from ub_j places
            possible_blocks = list()
            b_prime_sample = None
            b_lplus1_sample = None  # (14)

def get_unique_prob_block_location(model, thetaG, block_l, l, psi, xOut):
    if model['name'] == "mKPGM":
        # Calc U (set unique probabilities), use node attributes
        # For mKPGM it's just the theta[i][j] values
        # U = thetaG.flatten()
        U = [i for row in thetaG for i in row]

        # Index T by probability (pi_u)
        T = dict.fromkeys(U, dict.fromkeys(psi, list()))
        # T = dict.fromkeys(U, list())

        b = model['b']

        # get indices for edges (non-zero probability)
        for prob in block_l.keys():    # these correspond to theta values for mKPGM
            blocks = list(block_l[prob])  # TODO: how do I init this???
            for k in range(model['K'] - l - 1):  # TODO: this goes here?
                for s,t in blocks:
                    # map i,j from block[l] to u,v in E_OUT
                    # TODO: need to iterate over this K - l - 1 times to get vertices

                    blocks = [(s * b + i, t * b + j) for i in range(b) for j in range(b)]

                    for i in range(b):
                        for j in range(b):
                            u = s * b + i
                            v = t * b + j

                            block_loc = (u, v)

                            # descendent edge
                            edge_type = (xOut[u], xOut[v])

                            T[prob][edge_type].append(block_loc)

                            # T[prob].append(block_loc)

        return U, T
    else:
        # TODO: calc U and T for KPGM
        raise NotImplemented


def calc_correlation(edges, labels):
    '''
    Calculate the Pearson correlation across edges.

    :param edges: list of tuples containing node indices
    :param labels:
    :return:
    '''
    x = []
    y = []

    for (u,v) in edges:
        x.append(labels[u])
        y.append(labels[v])

    return float(pearsonr(x,y)[0])
