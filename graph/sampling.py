import random
import numpy as np
import itertools
from scipy.stats.stats import pearsonr
from kronecker import mKPGM as mKPGM


def graph_sampling(graphIn, xIn, model, epsilon, distribution, thetaG=None):
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

    # TODO (1) learn parameters
    if thetaG:
        thetaX = f_x(xIn, distribution)
        # psi = list(itertools.combinations_with_replacement(thetaX.keys(), 2))
        psi = [(0,0), (0,1), (1,0), (1,1)]
        # psi = list(itertools.product(thetaX.keys(), repeat=2))
        # beta = [0.25, 0.25, 0.25, 0.25]
        # beta = [0.4375, 0.125, 0.125, 0.4375]
        # beta = [0.45, 0.05, 0.05, 0.45]
        # beta = [0.48, 0.01, 0.01, 0.48]
        # beta = [0.97, 0.01, 0.01, 0.01]
        beta = [0.05, 0.45, 0.45, 0.05]
    else:
        psi, beta, thetaX, thetaG = learn_parameters(graphIn, xIn, model, distribution)

    n = pow(model['b'], model['K'])

    # (2) sample node attributes xOut from P(X|thetaX)
    # TODO: change to number of vertices in graphOut
    # xOut = sample_x(thetaX, distribution, len(verticesIn))
    xOut = sample_x(thetaX, distribution, n)

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
    verticesOut, edgesOut = maxent_edge_sampling(model, thetaG, graphOut.blocks[-1], psi, beta, xOut)

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


def maxent_edge_sampling(model, thetaG, block, psi, beta, xOut):
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
    U, T = get_unique_prob_edge_location(model, thetaG, block, psi, xOut)
    # N_e = 0
    Nus = []

    # for each unique prob. pi_u
    for pi_u in U:
        # Draw num. edges per unique prob.
        n_u = np.random.binomial(len(T[pi_u]), pi_u)
        Nus.append(n_u)
        # Accum. total num. edges to be sampled
        # N_e += n_u
    N_e = sum(Nus)

    # Draw num. edges per edge-type to match rho_IN
    gamma = list(np.random.multinomial(n=N_e, pvals=beta, size=1))
    # n = Number of experiments (int)
    # pvals = Probabilities of each of the p different outcomes (sequence of floats, length p)
    # size = Output shape (int or tuple of ints, optional)

    E_OUT = []
    for i, u in enumerate(U):
        # Draw num. edges per edge type for pi_u
        Y = list(np.random.multinomial(n=Nus[i], pvals=[np.double(g)/N_e for g in gamma[0]], size=1))
        for j, p in enumerate(psi):
            possible_edges = list(T[u][p])
            random.shuffle(possible_edges)
            edges = possible_edges[:Y[0][j]]

            E_OUT.extend(edges)
            gamma[0][j] -= Y[0][j]
            N_e -= Y[0][j]

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

        # get indices for edges (non-zero probability)
        for prob in block.keys():
            for i,j in block[prob]:
                edge_type = (xOut[i], xOut[j])
                edge_loc = (i, j)
                T[prob][edge_type].append(edge_loc)

        return U, T
    else:
        # TODO: calc U and T for KPGM
        raise NotImplemented


def lp_block_search(model, thetaG, blockSample_l, psi, beta, xOut):
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
    # TODO: LPBlockSearch
    raise NotImplemented


def get_unique_prob_block_location(model, thetaG, block_l):
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
