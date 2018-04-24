import random
# import math
import numpy as np
import itertools
from scipy.stats.stats import pearsonr
from scipy.optimize import linprog
from kronecker import mKPGM as mKPGM


def graph_sampling(graph_in, x_in, model, epsilon, distribution, params_test=None):
    """
    Graph Sampling algorithm

    :param graph_in: contains list of vertices and list of edges
    :type graph_in: tuple

    :param x_in: node attributes for graph_in
    :type x_in: list

    :param model: GNM and parameters
    :type model: dict

    :param epsilon: error
    :type epsilon: float

    :param distribution: "binomial" or "multinomial:
    :type distribution: String

    :param params_test: additional parameters for testing this code
    
    :return: graphOut (graph), x_out (attributes), rhoOut (correlation)
    :rtype: tuple, list, float
    """

    verticesIn, edgesIn = graph_in

    # model = ??? -- is this just b and l? ...since theta and K can be learned
    # theta_g=None, beta=None
    # TODO (3) learn parameters

    # TODO: testing code
    psi = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # psi = list(itertools.product(theta_x.keys(), repeat=2))

    if params_test:
        # if "beta" in params_test.keys():
        beta = params_test["beta"]

        # if "theta_g" in params_test.keys():
        theta_g = params_test["theta_g"]
        theta_x = f_x(x_in, distribution)

    else:
        psi, beta, theta_x, theta_g = learn_parameters(graph_in, x_in, model, distribution)

    if not params_test:
        psi, beta, theta_x, theta_g = learn_parameters(graph_in, x_in, model, distribution)
        n = pow(model['b'], model['K'])
        # (4) sample node attributes x_out from P(X|theta_x)
        x_out = sample_x(theta_x, distribution, n)
    else:
        # if "beta" in params_test.keys():
        beta = params_test["beta"]

        # if "theta_g" in params_test.keys():
        theta_g = params_test["theta_g"]
        theta_x = f_x(x_in, distribution)
        # TODO: add param to hardcode or sample
        x_out = x_in

    # theta_g = [[0.7, 0.4], [0.4, 0.5]]
    # graphOut = model.mKPGM(theta_g, K=5, b=2, l=2)

    # (5) init rhoOut = (correlation) and l_o = K - l - 1
    rhoOut = np.inf
    rhoIn = calc_correlation(edgesIn, x_in)

    if model['name'] == "mKPGM":
        # l_o = model['K'] - model['l'] - 1
        l_o = model['l'] - 1
    else:
        # TODO: implement for KPGM
        raise NotImplemented
    # K can be learned
    # l has to be specified
    # rho_OUT = np.inf
    # l_o = K - l - 1

    # Init graphOut
    # TODO: implement for KPGM
    # g = mKPGM.mKPGM(theta_g, model['K'], model['b'], model['l'])
    graphOut = None

    # (8) sample edges

    # TODO: (6) while loop --
    while (np.isnan(rhoOut) or np.isinf(abs(rhoOut - rhoIn))) and \
            abs(rhoOut - rhoIn) > epsilon and l_o >= 0:
        graphOut = mKPGM.mKPGM(theta_g, model['K'], len(theta_g[0]), model['l'])

        # TODO: (8-9) block sampling with LP
        # for l in range(l_o + 1, model['K'] - model['l'] - 1):
        for l in range(l_o + 1, model['l'] - 1):
            idx = l - 1 - model['l']
            last = lp_block_search(model, theta_g, graphOut.blocks[idx], model['l'], psi, beta, x_out)
        # (10) sample last block
        # TODO: testing version
        if params_test and "last_block" in params_test and params_test["last_block"]:
            verticesOut, edgesOut = maxent_edge_sampling(model, theta_g, params_test["last_block"], psi, beta, x_out)
        else:
            verticesOut, edgesOut = maxent_edge_sampling(model, theta_g, graphOut.blocks[-1], model['l'], psi, beta, x_out)

        # verticesOut, edgesOut = maxent_edge_sampling(model, theta_g, graphOut.blocks[-1], psi, beta, x_out)
        graphOut.edges = edgesOut

        # TODO: (11) calculate rhoOut
        # Initial version
        # TODO: use graphOut.edges
        rhoOut = calc_correlation(graphOut.edges, x_out)
        # rhoOut = calc_correlation(edgesIn, x_out)

        # TODO: (12) update l_o
        l_o = l_o - 1

    return graphOut, x_out
    # return graph_in, x_out


def learn_parameters(graph_in, x_in, model, distribution):
    """

    :param graph_in: contains list of vertices and edges
    :type graph_in: tuple

    :param x_in: node attributes for graph_in
    :type x_in: list

    :param model: GNM and parameters
    :type model: dict

    :param distribution: "binomial" or "multinomial"
    :type distribution: String

    :return: psi (edge types), beta (fraction of edge types),
    theta_x (parameters for P(X)), theta_g (parameters for P(G))
    :rtype: list of tuples, dictionary, dictionary, matrix
    """
    # TODO: (1) learn parameters (psi, beta, theta_x, theta_g)
    # TODO: do I need model in here??
    psi = None
    beta = None
    theta_x = f_x(x_in, distribution)
    theta_g = [[0.7, 0.4], [0.4, 0.5]]

    return psi, beta, theta_x, theta_g


def f_x(x_in, distribution):
    """
    function to learn theta_x parameters for P(X)

    :param x_in: attributes for vertices of graph_in
    :type x_in: list

    :param distribution: "binomial" or "multinomial:
    :type distribution: String

    :return:
    """
    # (1) learn parameters theta_x

    if distribution in ["binomial", "multinomial"]:
        labels = list(set(x_in))
        theta_x = {}

        for l in labels:
            theta_x[l] = float(x_in.count(l)) / len(x_in)

        return theta_x
    else:
        raise ValueError("Supported distributions are 'binomial' and 'multinomial'")


def sample_x(theta_x, distribution, num_samples):
    """
    Sample node attributes x_out from P(X|theta_x)

    :param theta_x: parameters for P(X)
    :type theta_x: dictionary

    :param distribution: "binomial" or "multinomial:
    :type distribution: String

    :param num_samples: number of samples
    :type num_samples: int

    :return: x_out (new attributes for graphOut)
    :rtype: list
    """

    if distribution in ["binomial", "multinomial"]:
        labels = theta_x.keys()
        probabilities = [theta_x[l] for l in labels]

        # tmp = np.random.multinomial(n=(len(labels) - 1), pvals=probabilities, size=num_samples)
        tmp = np.random.multinomial(n=1, pvals=probabilities, size=num_samples)
        x_out = [t[0] for t in tmp]

        return x_out
    else:
        raise ValueError("Supported distributions are 'binomial' and 'multinomial'")


def maxent_edge_sampling(model, theta_g, block, l, psi, beta, x_out):
    """

    :param model: GNM and parameters
    :type model: dict

    :param theta_g: parameters for marginal distribution of network structure P(G)
    :type theta_g: matrix

    :param block: sample block from penultimate iteration of mKPGM
    :type block: matrix

    :param psi: edge types
    :type psi: list

    :param beta: fraction of edges of each type
    :type beta: list

    :param x_out: node attributes
    :type x_out: list

    :return: graphOut (output graph, contains num. of vertices and edge list)
    :rtype: tuple
    """
    # U = unique probabilities
    # T = edge locations
    # (3)
    U, T = get_unique_prob_edge_location(model, theta_g, block, psi, x_out)
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


def get_unique_prob_edge_location(model, theta_g, block, psi, x_out):
    """

    :param model: GNM and parameters
    :type model: dict

    :param theta_g: parameters for marginal distribution of network structure P(G)
    :type theta_g: matrix

    :param block: sample block from penultimate iteration of mKPGM
    :type block: dict

    :param psi: edge types
    :type psi: list

    :param x_out: node attributes
    :type x_out: list

    :return: U (unique probabilities), T (edge locations)
    :rtype: set, matrix
    """

    if model['name'] == "mKPGM":
        # Calc U (set unique probabilities), use node attributes
        # For mKPGM it's just the theta[i][j] values
        # U = theta_g.flatten()
        U = [i for row in theta_g for i in row]

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

                        edge_type = (x_out[u], x_out[v])
                        edge_loc = (u, v)
                        T[prob][edge_type].append(edge_loc)

        return U, T
    else:
        # TODO: calc U and T for KPGM
        raise NotImplemented


def lp_block_search(model, theta_g, block_sample_l, l, psi, beta, x_out):
    """

    :param model: GNM and parameters
    :type model: dict

    :param theta_g: parameters for marginal distribution of network structure P(G)
    :type theta_g: matrix

    :param block_sample_l: sample block from l-th iteration of mKPGM
    :type block: matrix

    :param psi: edge types
    :type psi: list

    :param beta: fraction of edges of each type
    :type beta: list

    :param x_out: node attributes
    :type x_out: list

    :return:
        :blockSample_lPlus1: sampled block in l+1
    """

    # (3)
    U, T = get_unique_prob_block_location(model, theta_g, block_sample_l, l, psi, x_out)

    # for each unique prob. pi_u
    # (4)
    for u, pi_u in enumerate(U):
        # Draw num. blocks to sample per unique prob.
        T_u = [item for item in T[pi_u]]
        # T_u = [t_k[psi_j] for t_k in T[pi_u] for psi_j in psi]
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
        # A = [[len(t_k[psi_j]) for t_k in T_u] for psi_j in psi]
        A = determine_matrix_a(model, psi, T_u, l, x_out)

        # A = []
        # for psi_j in psi:
        #     A_j = []
        #     for t_k in T[pi_u]:
        #         A_jk = len(t_k[psi_j])
        #         A_j.append(A_jk)
        #     A.append(A_j)

        ub = []
        for k, t_k in enumerate(T_u):  # (9)
            ub_k = sum([A[j][k] for j in range(len(psi))])
            ub.append(ub_k)

        # (11) TODO: (REVISE) solve linear equation
        c = np.array([[-1 * item for item in row] for row in A])
        Aeq = np.ones((1, len(T_u)))
        beq = np.array([n_u])
        bounds = np.array([(0, ub_k) for ub_k in ub])
        chi = linprog(c=c, A_ub=np.array(A), b_ub=np.array(e),
                A_eq=Aeq, b_eq=beq, bounds=bounds,
                method='interior-point')


        # (12) TODO: sample block
        for k, t_k in enumerate(T_u):
            # (13) Sampling X_j blocks at random from ub_j places
            # t_k is one location from T_u
            possible_blocks = list(t_k)
            b_prime_sample = None
            b_lplus1_sample = None  # (14)
    # TODO: return blocks, this is just placeholder for now
    return (None, None)


def determine_matrix_a(model, psi, T_u, l, x_out):
    # init A to all zeros
    # A is |phi| x |T_u| (originally)
    a = np.zeros((len(psi), len(T_u)))
    b = model['b']
    cnt_edges = []

    for k, t_k in enumerate(T_u):
        s, t = t_k
        prev_level = [(s, t)]
        curr_edges = []

        # calculate edges
        for itr in range(model['K'] - l):
            for s, t in prev_level:
                curr_edges = [(s * b + i, t * b + j) for i in range(b) for j in range(b)]
            prev_level = list(curr_edges)

        # iterate over edges, get edge types and add to A
        for u, v in curr_edges:
            # descendant edge
            edge_type = (x_out[u], x_out[v])
            j = psi.index(edge_type)
            a[j][k] += 1

        c_k = [a[j][k] for j in range(len(psi))]
        cnt_edges.append(c_k)

    # Reverse lookup for configurations
    configs = dict.fromkeys(cnt_edges, list())
    for k, c_k in enumerate(cnt_edges):
        configs[c_k].append(T_u[k])


    # Create a list of lists, then convert to set

    unique_a_transp = unique_rows(a.T)
    
    return unique_a_transp.T, configs


def get_unique_prob_block_location(model, theta_g, block_l, l, psi, x_out):
    if model['name'] == "mKPGM":
        # Calc U (set unique probabilities), use node attributes
        # For mKPGM it's just the theta[i][j] values
        # U = theta_g.flatten()
        U = [i for row in theta_g for i in row]

        # Index T by probability (pi_u)
        # T = dict.fromkeys(U, dict.fromkeys(psi, list()))
        T = dict.fromkeys(U, list())

        b = model['b']

        # get indices for edges (non-zero probability)
        for prob in block_l.keys():    # these correspond to theta values for mKPGM
            # blocks = list(block_l[prob])  # TODO: how do I init this???
            # for k in range(model['K'] - l - 1):  # TODO: this goes here?
            #     for s,t in blocks:
            for s,t in block_l[prob]:
                # map i,j from block[l] to u,v in E_OUT
                # Don't need to iterate over this K - l - 1 times to get vertices

                # blocks = [(s * b + i, t * b + j) for i in range(b) for j in range(b)]

                for i in range(b):
                    for j in range(b):
                        u = s * b + i
                        v = t * b + j

                        block_loc = (u, v)

                        # descendant edge
                        # edge_type = (x_out[u], x_out[v])
                        # T[prob][edge_type].append(block_loc)
                        T[prob].append(block_loc)

        return U, T
    else:
        # TODO: calc U and T for KPGM
        raise NotImplemented


def unique_rows(a):
    # https://stackoverflow.com/a/8567929/6846164
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


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
