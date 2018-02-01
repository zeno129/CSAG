import numpy as np
from scipy.stats.stats import pearsonr
from kronecker import mKPGM as model


def graph_sampling(graphIn, xIn, model, epsilon, f_x, sample_x):
    """
    Graph Sampling algorithm

    :param graphIn: tuple with set of vertices and edges
    :param xIn: node attributes for graphIn
    :param model: generative network model
    :param epsilon: error
    :param f_x: function to learn thetaX parameters for P(X)
    :param sample_x: function to sample from P(X|theta^X)
    :return:
        :graphOut: graph
        :xOut: attributes
        :rhoOut: correlation
    """

    verticesIn, edgesIn = graphIn

    # model = ??? -- is this just b and l? ...since theta and K can be learned

    # (1) learn parameters
    phi, beta, thetaX, thetaG = learn_parameters(graphIn, xIn, model, f_x)


    # (2) sample node attributes xOut from P(X|theta^X)
    xOut = sample_x(thetaX)

    # thetaG = [[0.7, 0.4], [0.4, 0.5]]
    # graphOut = model.mKPGM(thetaG, K=5, b=2, l=2)


    # TODO: (3) init rhoOut = (correlation) and l_o = K - l - 1
    # K can be learned
    # l has to be specified
    # rho_OUT = np.inf
    # l_o = K - l - 1
    # TODO: (4) while loop --
    # TODO: (5-8) block sampling with ME and LP


    # TODO: (9) calculate rhoOut
    # Initial version
    # rhoOut = calc_correlation(graphOut.edges, xOut)
    rhoOut = calc_correlation(edgesIn, xOut)


    # TODO: (10) update l_o

    # return graphOut, xOut
    return graphIn, xOut


def learn_parameters(graphIn, xIn, model, f_x):
    """

    :param graphIn: tuple with set of vertices and edges
    :param xIn: node attributes for graphIn
    :param model: generative network model
    :return:
        :phi: list of tuples with edge types
                (e.g., (0,0), (0,1), (1,1))
        :beta: dictionary with fraction of edges of each type
        :thetaX: parameters for marginal distribution of node attributes P(X);
                   Ex. MLE for Bernoulli trials
        :thetaG: parameters for marginal distribution of network structure P(G);
                   Ex. mKPGM parameters
    """
    # TODO: (1) learn parameters (phi, beta, thetaX, thetaG)
    phi = None
    beta = None
    thetaX = f_x(xIn)
    thetaG = [[0.7, 0.4], [0.4, 0.5]]

    return phi, beta, thetaX, thetaG


def ME_edge_sampling(model, theta, psy, beta):
    pass

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
