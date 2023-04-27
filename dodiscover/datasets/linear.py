from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _sample_from_graph(
    G,
    top_sort_idx,
    rng: np.random.Generator,
) -> Dict:
    """Private function to sample a single iid sample from a graph for all nodes.

    Used to parallelize the sampling of multiple iid samples from a graph.
    See `make_linear_gaussian` for more details.

    Returns
    -------
    nodes_sample : dict
        The sample per node.
    """
    nodes_sample = dict()

    for node_idx in top_sort_idx:
        # get all parents
        parents = sorted(list(G.predecessors(node_idx)))

        # sample noise
        mean = G.nodes[node_idx]["gaussian_noise_function"]["mean"]
        std = G.nodes[node_idx]["gaussian_noise_function"]["std"]
        node_noise = rng.normal(loc=mean, scale=std)
        node_sample = 0

        # sample weight and edge function for each parent
        for parent in parents:
            weight = G.nodes[node_idx]["parent_functions"][parent]["weight"]
            func = G.nodes[node_idx]["parent_functions"][parent]["func"]
            node_sample += weight * func(parent)

        # set the node attribute "functions" to hold the weight and function wrt each parent
        node_sample += node_noise
        nodes_sample[node_idx] = node_sample
    return nodes_sample


def sample_from_graph(
    G: nx.DiGraph, n_samples: int = 1000, n_jobs: Optional[int] = None, random_state=None
):
    """Sample a dataset from a linear Gaussian graph.

    Parameters
    ----------
    G : nx.DiGraph
        A linear DAG from which to sample. Must have been set up with
        :func:`pywhy_graphs.functional.make_graph_linear_gaussian`.
    n_samples : int, optional
        Number of samples to generate, by default 1000.
    n_jobs : Optional[int], optional
        Number of jobs to run in parallel, by default None.
    random_state : int, optional
        Random seed, by default None.

    Returns
    -------
    data : pd.DataFrame of shape (n_samples, n_nodes)
        A pandas DataFrame with the iid samples.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The input graph must be a DAG.")
    if not G.graph.get("linear_gaussian", True):
        raise ValueError("The input graph must be a linear Gaussian graph.")

    rng = np.random.default_rng(random_state)

    # Create list of topologically sorted nodes
    top_sort_idx = list(nx.topological_sort(G))

    # Sample from graph
    if n_jobs is None:
        data = []
        for _ in range(n_samples):
            node_samples = _sample_from_graph(
                G,
                top_sort_idx,
                rng,
            )
            data.append(node_samples)
        data = pd.DataFrame.from_records(data)
    else:
        out = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_sample_from_graph)(
                G,
                top_sort_idx,
                rng,
            )
            for _ in range(n_samples)
        )
        data = pd.DataFrame.from_records(out)

    return data
