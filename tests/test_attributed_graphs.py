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

    _, attributes = sampling.graph_sampling((range(g.vertices), g.edges), None, None, None)

    assert len(attributes) == g.vertices
