# import itertools
# import operator
import numpy as np
from kronecker import mKPGM as model
from graph import sampling


def test_graph_sampling_binomial():
    """
    Test dumb version:
    Create graph and get random attributes back.
    Use binomial distribution.
    """
    b = 2
    k = 5
    l = 2
    theta = [[0.7, 0.4], [0.4, 0.5]]
    g = model.mKPGM(theta, k, b, l)

    x = list(np.random.random_integers(low=0, high=1, size=g.vertices))

    _, attributes = sampling.graph_sampling(graphIn=(range(g.vertices), g.edges),
                                            xIn=x,
                                            model=None,
                                            epsilon=0.0,
                                            distribution="binomial")

    # TODO: calculate with graphOut
    n = g.vertices
    p = 0.5
    mean = n * p
    variance = mean * (1 - p)

    assert len(attributes) == g.vertices
    assert attributes.count(1) <= mean + variance
    assert attributes.count(1) >= mean - variance


def test_graph_sampling_multinomial():
    """
    Test dumb version:
    Create graph and get random attributes back.
    Use binomial distribution.
    """
    b = 2
    k = 5
    l = 2
    theta = [[0.7, 0.4], [0.4, 0.5]]
    g = model.mKPGM(theta, k, b, l)

    x = list(np.random.random_integers(low=0, high=3, size=g.vertices))

    _, attributes = sampling.graph_sampling(graphIn=(range(g.vertices), g.edges),
                                            xIn=x,
                                            model=None,
                                            epsilon=0.0,
                                            distribution="multinomial")

    # TODO: calculate with graphOut
    n = g.vertices
    p = 0.25
    mean = n * p
    variance = mean * (1 - p)

    assert len(attributes) == g.vertices
    assert attributes.count(1) <= mean + variance
    assert attributes.count(1) >= mean - variance

