from itertools import permutations

import bnlearn
import networkx as nx
import numpy as np
import pandas as pd
import pooch
import pytest
import pywhy_graphs as pgraphs
from pywhy_graphs import AugmentedPAG, AugmentedPAG
from pywhy_graphs.export import numpy_to_graph

from dodiscover import SFCI, InterventionalContextBuilder, make_context
from dodiscover.ci import GSquareCITest, Oracle
from dodiscover.constraint.utils import dummy_sample

from .test_psifcialg import Test_PsiFCI

np.random.seed(12345)


class Test_SFCI(Test_PsiFCI):
    def setup_method(self):
        # construct a causal graph that will result in
        # x -> y <- z
        G = nx.DiGraph([("x", "y"), ("z", "y")])
        oracle = Oracle(G)
        alg = SFCI(ci_estimator=oracle, cd_estimator=oracle)

        self.context_func = lambda: make_context(create_using=InterventionalContextBuilder)
        self.G = G
        self.ci_estimator = oracle
        self.alg = alg

    def test_basic_chain_graph(self):
        directed_edges = [("x", "y")]

        graph = pgraphs.AugmentedGraph(
            incoming_directed_edges=directed_edges,
        )
        graph.add_f_node({"x"})
        graph.add_f_node({"x"}, require_unique=False)
        graph.add_s_node((1, 2), {"y"})
