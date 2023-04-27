from typing import Callable, Dict, List, Optional

import numpy as np
from pywhy_graphs.functional import make_graph_linear_gaussian
from pywhy_graphs.simulate import simulate_random_er_dag

from .linear import sample_from_graph


def make_multidomain_er(
    n_domains: int = 2,
    s_nodes: Dict[int, List] = None,
    n_nodes: int = 10,
    p: float = 0.5,
    n_samples=100,
    node_mean_lims: Optional[List[float]] = None,
    node_std_lims: Optional[List[float]] = None,
    edge_functions: List[Callable[[float], float]] = None,
    edge_weight_lims: Optional[List[float]] = None,
    n_jobs: Optional[int] = None,
    random_state=None,
):
    """Make a multi-domain ER graph and sample dataset.

    Parameters
    ----------
    n_domains : int, optional
        The number of domains to generate, by default 2.
    s_nodes : Dict[int, List], optional
        Dictionary of singleton nodes for each domain with keys being domain indices.
        By default None, which samples a random singleton node to be different in the target relative to each domain.
    n_nodes : int, optional
        The number of nodes in the graph, by default 10.
    p : float, optional
        The edge probability for the random graph, by default 0.5.
    n_samples : int, optional
        The number of samples to generate for each domain, by default 100.
    node_mean_lims : Optional[List[float]], optional
        The limits for the node mean values, a list of length 2, by default None.
    node_std_lims : Optional[List[float]], optional
        The limits for the node standard deviation values, a list of length 2, by default None.
    edge_functions : List[Callable[[float], float]], optional
        List of functions to use for edge weights, by default None.
    edge_weight_lims : Optional[List[float]], optional
        The limits for the edge weight values, a list of length 2, by default None.
    n_jobs : Optional[int], optional
        The number of jobs to run in parallel, by default None.
    random_state : _type_, optional
        Seed for the random number generator, by default None.

    Returns
    -------
    G : nx.DiGraph
        The selection diagram graph.
    datasets : Dict of List of pd.DataFrame
        The datasets for each domain, keyed by index corresponding to the domains.
        The target is the last index (highest index) by convention.
    """
    rng = np.random.default_rng(seed=random_state)
    if s_nodes is None:
        # sample a random singleton node to be different in the target relative to each domain
        # the last domain is the target by convention
        s_nodes = {idx: [rng.uniform(n_nodes)] for idx in range(0, n_domains - 1)}

    # first simulate the random ER graph
    G = simulate_random_er_dag(n_nodes, p, seed=random_state)

    # from the graph, we will instantiate a data-generating model
    G = make_graph_linear_gaussian(
        G,
        node_mean_lims=node_mean_lims,
        node_std_lims=node_std_lims,
        edge_functions=edge_functions,
        edge_weight_lims=edge_weight_lims,
        random_state=random_state,
    )

    # now, we will sample the target domain observational distribution
    data = sample_from_graph(G, n_samples=n_samples, n_jobs=n_jobs, random_state=random_state)

    datasets = []
    for idx in range(n_domains - 1):
        # make sure we don't modify the original graph
        G_copy = G.copy()

        # now, we will sample the other domain observational distributions
        this_domain_snodes = s_nodes[idx]

        # modify the noise properties, or perturb the function at those s-nodes
        # TODO: decide on a perturbation policy to apply
        for s_node in this_domain_snodes:
            G_copy.nodes[s_node]["gaussian_noise_function"]["mean"] = rng.uniform(-1, 1)
            G_copy.nodes[s_node]["gaussian_noise_function"]["std"] = rng.uniform(-1, 1)

        # now sample a dataset from the graph again
        source_data = sample_from_graph(
            G_copy, n_samples=n_samples, n_jobs=n_jobs, random_state=random_state
        )
        datasets.append(source_data)
    datasets.append(data)

    # now modify the existing graph to have the S-nodes
    for idx in range(n_domains - 1):
        this_domain_snodes = s_nodes[idx]

        # add a new node to the graph for each s-node in this domain
        for s_node in this_domain_snodes:
            G.add_node(f"S-{s_node}", s_node=True, domain=idx, target=n_domains)

    return G, datasets
