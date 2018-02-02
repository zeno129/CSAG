import numpy as np
from scipy.stats.stats import pearsonr
from kronecker import mKPGM as model


def graph_sampling(graphIn, xIn, model, epsilon, distribution):
    """
    Graph Sampling algorithm

    :param graphIn: contains list of vertices and list of edges
    :type graphIn: tuple

    :param xIn: node attributes for graphIn
    :type xIn: list

    :param model: generative network model
    :type model:

    :param epsilon: error
    :type epsilon: float

    :param distribution: "binomial" or "multinomial:
    :type distribution: String
    
    :return: graphOut (graph), xOut (attributes), rhoOut (correlation)
    :rtype: tuple, list, float
    """

    verticesIn, edgesIn = graphIn

    # model = ??? -- is this just b and l? ...since theta and K can be learned

    # (1) learn parameters
    phi, beta, thetaX, thetaG = learn_parameters(graphIn, xIn, model, distribution)


    # (2) sample node attributes xOut from P(X|thetaX)
    # TODO: change to number of vertices in graphOut
    xOut = sample_x(thetaX, distribution, len(verticesIn))

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
    # TODO: use graphOut.edges
    # rhoOut = calc_correlation(graphOut.edges, xOut)
    rhoOut = calc_correlation(edgesIn, xOut)


    # TODO: (10) update l_o

    # return graphOut, xOut
    return graphIn, xOut


def learn_parameters(graphIn, xIn, model, distribution):
    """

    :param graphIn: contains list of vertices and edges
    :type graphIn: tuple

    :param xIn: node attributes for graphIn
    :type xIn: list

    :param model: generative network model

    :param distribution: "binomial" or "multinomial"
    :type distribution: String

    :return: phi (edge types), beta (fraction of edge types),
    thetaX (parameters for P(X)), thetaG (parameters for P(G))
    :rtype: list of tuples, dictionary, dictionary, matrix
    """
    # TODO: (1) learn parameters (phi, beta, thetaX, thetaG)
    phi = None
    beta = None
    thetaX = f_x(xIn, distribution)
    thetaG = [[0.7, 0.4], [0.4, 0.5]]

    return phi, beta, thetaX, thetaG


def f_x(xIn, distribution):
    """
    function to learn thetaX parameters for P(X)

    :param xIn: attributes for vertices of graphIn
    :type xIn: list

    :param distribution: "binomial" or "multinomial:
    :type distribution: String

    :return:
    """
    # TODO: (1) learn parameters thetaX

    if distribution in ["binomial", "multinomial"]:
        labels = list(set(xIn))
        thetaX = {}

        for l in labels:
            thetaX[l] = float(xIn.count(l)) / len(xIn)

        # if distribution == "binomial":
        #     for l in labels:
        #         thetaX[l] = float(xIn.count(l)/len(xIn))
        # elif distribution == "multinomial":
        #     pass

        # return {"low": 2, "size": xIn}
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

        tmp = np.random.multinomial(n=(len(labels) - 1), pvals=probabilities, size=num_samples)
        xOut = [t[0] for t in tmp]

        return xOut
    else:
        raise ValueError("Supported distributions are 'binomial' and 'multinomial'")


def ME_edge_sampling(model, thetaG, phi, beta):
    """

    :param model: generative network model
    :param thetaG: parameters for marginal distribution of network structure P(G)
    :param phi: list of tuples with edge types
    :param beta: dictionary with fraction of edges of each type
    :return:
        :graphOut: output graph; tuple (num of vertices, list of edge tuples)
    """
    # TODO: ME_edge_sampling
    pass


def LPBlockSearch(model, thetaG, blockSample_l, phi, beta):
    """

    :param model: generative network model
    :param thetaG: parameters for marginal distribution of network structure P(G)
    :param blockSample_l: matrix with block probabilities?
    :param phi: list of tuples with edge types
    :param beta: dictionary with fraction of edges of each type
    :return:
        :blockSample_lPlus1: sampled blocks in l+1
    """
    # TODO: LPBlockSearch
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
