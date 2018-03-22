# import itertools
# import operator
import random
import numpy as np
from kronecker import mKPGM as model
from graph import sampling
from scipy.stats.stats import pearsonr


# def test_graph_sampling_binomial():
#     """
#     Test dumb version:
#     Create graph and get random attributes back.
#     Use binomial distribution.
#     """
#     b = 2
#     k = 5
#     l = 2
#     theta = [[0.7, 0.4], [0.4, 0.5]]
#     g = model.mKPGM(theta, k, b, l)
#     mymodel = {'name': "KPGM", 'k': k, 'b': b, 'l': l}
#
#     x = list(np.random.random_integers(low=0, high=1, size=g.vertices))
#
#     _, attributes = sampling.graph_sampling(graphIn=(range(g.vertices), g.edges),
#                                             xIn=x,
#                                             model=mymodel,
#                                             epsilon=0.0,
#                                             distribution="binomial")
#
#     # TODO: calculate with graphOut
#     n = g.vertices
#     p = 0.5
#     mean = n * p
#     variance = mean * (1 - p)
#
#     assert len(attributes) == g.vertices
#     assert attributes.count(1) <= mean + variance
#     assert attributes.count(1) >= mean - variance
#
#
# def test_graph_sampling_multinomial():
#     """
#     Test dumb version:
#     Create graph and get random attributes back.
#     Use binomial distribution.
#     """
#     b = 2
#     k = 5
#     l = 2
#     theta = [[0.7, 0.4], [0.4, 0.5]]
#     g = model.mKPGM(theta, k, b, l)
#     mymodel = {'name': "KPGM", 'k': k, 'b': b, 'l': l}
#
#     x = list(np.random.random_integers(low=0, high=3, size=g.vertices))
#
#     _, attributes = sampling.graph_sampling(graphIn=(range(g.vertices), g.edges),
#                                             xIn=x,
#                                             model=mymodel,
#                                             epsilon=0.0,
#                                             distribution="multinomial")
#
#     # TODO: calculate with graphOut
#     n = g.vertices
#     p = 0.25
#     mean = n * p
#     variance = mean * (1 - p)
#
#     assert len(attributes) == g.vertices
#     assert attributes.count(1) <= mean + variance
#     assert attributes.count(1) >= mean - variance


# def test_maxent_edge_sampling():
#     # model = "mKPGM"
#     mymodel = {'name': "KPGM", 'k': None, 'b': None, 'l': None}
#     theta = [0.7, 0.4, 0.4, 0.5]
#     num_nodes = 2 ** 4
#
#     # TODO: get real block from mKPGM model
#     block = [ np.random.choice(theta, num_nodes) for i in range(num_nodes)]
#     psi = [(0,0), (0,1), (1,0), (1,1)]
#     p = 0.25
#     probs = [p,p,p,p]
#     tmp = np.random.multinomial(n=1, pvals=probs, size=num_nodes)
#     xOut = [t[0] for t in tmp]
#
#     vertices, edges = sampling.maxent_edge_sampling(mymodel,theta,block,psi,probs,xOut)
#     edge_labels = [(xOut[u], xOut[v]) for u,v in edges]
#
#     # TODO: test for all probabilities
#     n = vertices
#     mean = n * p
#     variance = mean * (1 - p)
#
#     assert vertices == num_nodes
#     assert edge_labels.count(psi[0]) <= mean + variance
#     assert edge_labels.count(psi[0]) >= mean - variance


def test_graph_sampling_binomial_no_learning():
    """
    Test dumb version:
    Create graph and get random attributes back.
    Use binomial distribution.
    """
    b = 2
    k = 3
    # k = 10
    l = 2
    theta = [[0.7, 0.4], [0.4, 0.5]]
    n = pow(b, k)

    # g = model.mKPGM(theta, k, b, l)
    mymodel = {'name': "mKPGM", 'K': k, 'b': b, 'l': l, 'theta': theta}
    keys = [t for row in theta for t in row]
    last_block = dict.fromkeys(keys, [])
    last_block[keys[0]] = [(0,2), (2,0), (2,6), (4,6), (6,0), (6,4)]
    last_block[keys[1]] = [(0,3), (2,1), (2,7), (4,7), (6,1), (6,5)]
    last_block[keys[2]] = [(1,2), (3,0), (3,6), (5,6), (7,0), (7,4)]
    last_block[keys[3]] = [(1,3), (3,1), (3,7), (5,7), (7,1), (7,5)]

    # x = list(np.random.random_integers(low=0, high=1, size=n))

    x = [0] * (n / 2)
    x.extend([1] * (n / 2))
    # random.shuffle(x)

    # TODO: specify beta directly
    # beta = fraction of edges of each type
    tries = [[0.25, 0.25, 0.25, 0.25],
             # [0.4375, 0.125, 0.125, 0.4375],
             [0.45, 0.05, 0.05, 0.45],
             [0.48, 0.01, 0.01, 0.48],
             [0.97, 0.01, 0.01, 0.01],
             [0.05, 0.45, 0.45, 0.05]]
    for beta in tries:
        params_test = {"beta": beta, "thetaG": theta, "last_block": last_block}
        graphOut, xOut = sampling.graph_sampling(graphIn=(None, None),
                                                xIn=x,
                                                model=mymodel,
                                                epsilon=0.0,
                                                distribution="binomial",
                                                params_test=params_test)

        with open("correlations.txt", "a") as myfile:
            corr = calc_correlation(graphOut.edges, xOut)
            myfile.write("beta = {}\t\tcorrelation = {}\n\n".format(beta, corr))

    # TODO: calculate with graphOut
    p = 0.5
    mean = n * p
    variance = mean * (1 - p)

    # TODO: figure out how to test correlation
    assert len(xOut) == n

    # assert calc_correlation(graphOut[1], xOut) == 0.5
    assert xOut.count(1) <= mean + variance
    assert xOut.count(1) >= mean - variance


# def test_multiple_random_graphs():
#     b = 2
#     # k = 5
#     k = 10
#     l = 2
#     n = pow(b, k)
#
#     probs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]  # v2
#     # probs = [0.7, 0.5, 0.3]  # v2
#     # probs = [0.5, 0.4, 0.3]  # v3
#     for p in probs:
#         theta = [[p, p], [p, p]]
#
#         mymodel = {'name': "mKPGM", 'K': k, 'b': b, 'l': l, 'theta': theta}
#
#         # x = list(np.random.random_integers(low=0, high=1, size=n))
#         x = [0] * (n/2)
#         x.extend([1] * (n/2))
#         random.shuffle(x)
#
#         graphOut, xOut = sampling.graph_sampling(graphIn=(None, None),
#                                                  xIn=x,
#                                                  model=mymodel,
#                                                  epsilon=0.0,
#                                                  distribution="binomial",
#                                                  thetaG=theta)
#
#         x_prob = 0.5
#         mean = n * x_prob
#         variance = n * x_prob * (1 - x_prob)
#
#         # TODO: figure out how to test correlation
#         assert len(xOut) == n
#         # assert calc_correlation(graphOut[1], xOut) == 0.5
#
#         # with open("correlations.txt", "a") as myfile:
#         #     corr = calc_correlation(graphOut.edges, xOut)
#         #     myfile.write("theta = {}\t\tcorrelation = {}\n\n".format(theta, corr))
#
#         assert xOut.count(1) <= mean + variance
#         assert xOut.count(1) >= mean - variance


# def test_range_of_graphs():
#     b = 2
#     # k = 5
#     k = 10
#     l = 2
#     n = pow(b, k)
#
#     theta_11 = [0.99, 0.95, 0.9, 0.85, 0.8]
#     theta_12 = [0.55, 0.45, 0.35, 0.25, 0.15]
#     theta_22 = [0.75, 0.65, 0.55, 0.45, 0.35]
#     # % theta_22 = [0.75 0.7 0.65 0.6 0.55];


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
