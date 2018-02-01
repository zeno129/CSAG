# import itertools
# import operator
import numpy as np
from kronecker import mKPGM as model
from graph import sampling


def test_graph_sampling():
    """
    Test dumb version:
    Create graph and get random attributes back
    :return:
    """
    b = 2
    k = 5
    l = 2
    theta = [[0.7, 0.4], [0.4, 0.5]]
    g = model.mKPGM(theta, k, b, l)

    _, attributes = sampling.graph_sampling((range(g.vertices), g.edges), g.vertices, None, None, f_x, sample_x)

    assert len(attributes) == g.vertices


def f_x(xIn):
    # return (2, xIn)
    return {"low": 2, "size": xIn}


def sample_x(thetaX):
    """
    Sample node attributes xOut from P(X|theta^X)

    :param n: num of attributes to generate
    :return:
    """
    # TODO: (2) sample node attributes xOut from P(X|theta^X)

    # Initial version (random)
    # binary class --> Bernoulli trials, p=0.5
    # xOut = np.random.randint(2, size=len(verticesIn))
    xOut = np.random.randint(**thetaX)

    return xOut