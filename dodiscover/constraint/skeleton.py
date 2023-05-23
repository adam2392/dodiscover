import logging
from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dodiscover.cd import BaseConditionalDiscrepancyTest
from dodiscover.ci import BaseConditionalIndependenceTest
from dodiscover.constraint.config import ConditioningSetSelection
from dodiscover.constraint.utils import is_in_sep_set
from dodiscover.typing import Column, SeparatingSet

from .._protocol import EquivalenceClass, Graph
from ..context import Context
from ..context_builder import ContextBuilder, InterventionalContextBuilder, make_context

logger = logging.getLogger()


def _parallel_test_xy_edges(
    parallel_fun: Callable[
        [pd.DataFrame, Callable, Column, Column, Optional[Set[Column]]], Tuple[float, float]
    ],
    conditional_test_func: Callable[[Column, Column, Optional[Set[Column]]], Tuple[float, float]],
    x_var: Column,
    y_var: Column,
    alpha: float,
    size_cond_set: int,
    max_combinations: Optional[int],
    possible_variables: Set[Column],
    data: pd.DataFrame,
    context: Context,
    cross_distribution_test: bool = False,
) -> Dict[str, Any]:
    """Private function used to test edge between X and Y in parallel for candidate separating sets.

    Parameters
    ----------
    conditional_test_func : Callable
        Conditional test function.
    x_var : Column
        The 'X' variable name.
    y_var : Column
        The 'Y' variable name.
    alpha : float
        The significance level for the conditional independence test.
    size_cond_set : int
        The current size of the conditioning set. This value will then generate
        ``(N choose 'size_cond_set')`` sets of candidate separating sets to test, where
        ``N`` is the size of 'possible_variables'.
    max_combinations : int
        The maximum number of conditional independence tests to run from the set
        of possible conditioning sets.
    possible_variables : Set[Column]
        A set of variables that are candidates for the conditioning set.
    data : pandas.Dataframe
        The dataset with variables as columns and samples as rows.

    Returns
    -------
    test_stat : float
        Test statistic.
    pvalue : float
        Pvalue.
    """
    prev_pvalue = 0.0

    # generate iterator through the conditioning sets
    conditioning_sets = _iter_conditioning_set(
        possible_variables=possible_variables,
        x_var=x_var,
        y_var=y_var,
        size_cond_set=size_cond_set,
    )

    # now iterate through the possible parents
    for comb_idx, cond_set in enumerate(conditioning_sets):
        # check the number of combinations of possible parents we have tried
        # to use as a separating set
        if max_combinations is not None and comb_idx >= max_combinations:
            break

        # either process within-distribution, or across distributions
        this_data = data
        if cross_distribution_test:
            # compute conditional independence test
            # get the sigma-map for this F-node
            distribution_idx = context.sigma_map[x_var]

            # get the distributions across the two distributions
            data_i = data[distribution_idx[0]].copy()
            data_j = data[distribution_idx[1]].copy()

            # name the group column the F-node, so Oracle works as expected
            data_i[x_var] = 0
            data_j[x_var] = 1
            this_data = pd.concat((data_i, data_j), axis=0)

        try:
            # compute conditional independence test
            test_stat, pvalue = parallel_fun(
                this_data, conditional_test_func, x_var, y_var, set(cond_set)
            )
        except Exception as e:
            print(e)
            test_stat = np.inf
            pvalue = 0.0

        # if any "independence" is found through inability to reject
        # the null hypothesis, then we will break the loop comparing X and Y
        # and say X and Y are conditionally independent given 'cond_set'
        if pvalue > alpha:
            break
        else:
            pvalue = max(pvalue, prev_pvalue)

    result: Dict[str, Any] = dict()
    result["x_var"] = x_var
    result["y_var"] = y_var
    result["cond_set"] = list(cond_set)
    result["test_stat"] = test_stat
    result["pvalue"] = pvalue
    return result


def candidate_cond_sets(
    condsel_method: ConditioningSetSelection,
    context: Context,
    x_var: Column,
    y_var: Column,
    keep_sorted: bool = False,
):
    """Compute candidate conditioning set using a specific method between two variables.

    Parameters
    ----------
    condsel_method : ConditioningSetSelection
        Method to compute candidate conditioning set.
    context : Context
        Causal context object with the graph and other information.
    x_var : Column
        The starting node.
    y_var : Column
        The ending node.
    keep_sorted : bool, optional
        Whether or not to keep the conditioning set sorted based on the context, by default False.

    Returns
    -------
    possible_variables : Set[Column]
        A set of variables that are candidates for the conditioning set.

    Notes
    -----
    The possible variables are determined by the method used to compute the candidate
    conditioning set. For example:
     - if the method is 'complete', then all variables in the graph are possible candidates.
     - if the method is 'neighbors', then only the neighbors of the starting node are possible
       candidates.
     - if the method is 'neighbors_path', then only the neighbors of the starting node that are
       also along a path to the ending node are possible candidates.
     - if the method is 'pds', then the possible candidates are determined by the
       PAG that computes the possibly d-separating set.
     - if the method is 'pds_path', then the possible candidates are determined by the
       PAG that computes the possibly d-separating set, but only those that are along a path to the
       ending node are possible candidates.
    """
    if condsel_method == ConditioningSetSelection.COMPLETE:
        possible_variables = set(context.init_graph.nodes)
    elif condsel_method == ConditioningSetSelection.NBRS:
        possible_variables = set(context.init_graph.neighbors(x_var))
        # possible_adjacencies.copy()
    elif condsel_method == ConditioningSetSelection.NBRS_PATH:
        # constrain adjacency set to ones with a path from x_var to y_var
        possible_variables = _find_neighbors_along_path(context.init_graph, start=x_var, end=y_var)
    elif condsel_method == ConditioningSetSelection.PDS:
        import pywhy_graphs as pgraph

        pag = context.state_variable("PAG", on_missing="ignore")
        max_path_length = context.state_variable("max_path_length")

        # determine how we want to construct the candidates for separating nodes
        # perform conditioning independence testing on all combinations
        possible_variables = pgraph.pds(
            pag, x_var, y_var, max_path_length=max_path_length  # type: ignore
        )
    elif condsel_method == ConditioningSetSelection.PDS_PATH:
        import pywhy_graphs as pgraph

        pag = context.state_variable("PAG", on_missing="ignore")
        max_path_length = context.state_variable("max_path_length")

        # determine how we want to construct the candidates for separating nodes
        # perform conditioning independence testing on all combinations
        possible_variables = pgraph.pds_path(
            pag, x_var, y_var, max_path_length=max_path_length  # type: ignore
        )

    if keep_sorted:
        # Note it is assumed in public API that 'test_stat' is set
        # inside the adj_graph
        possible_variables = sorted(
            possible_variables,
            key=lambda n: context.init_graph.edges[x_var, n]["test_stat"],
            reverse=True,
        )  # type: ignore

    if x_var in possible_variables:
        possible_variables.remove(x_var)
    if y_var in possible_variables:
        possible_variables.remove(y_var)

    return possible_variables



def _iter_conditioning_set(
    possible_variables: Iterable,
    x_var: Column,
    y_var: Column,
    size_cond_set: int,
) -> Iterable[Set]:
    """Iterate function to generate the conditioning set.

    Parameters
    ----------
    possible_variables : iterable
        A set/list/dict of possible variables to consider for the conditioning set.
        This can be for example, the current adjacencies.
    x_var : node
        The node for the 'x' variable.
    y_var : node
        The node for the 'y' variable.
    size_cond_set : int
        The size of the conditioning set to consider. If there are
        less adjacent variables than this number, then all variables will be in the
        conditioning set.

    Yields
    ------
    Z : set
        The set of variables for the conditioning set.
    """
    exclusion_set = {x_var, y_var}

    all_adj_excl_current = [p for p in possible_variables if p not in exclusion_set]

    # loop through all possible combinations of the conditioning set size
    for cond in combinations(all_adj_excl_current, size_cond_set):
        cond_set = set(cond)
        yield cond_set


def _find_neighbors_along_path(G: nx.Graph, start, end) -> Set:
    """Find neighbors that are along a path from start to end.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    start : Node
        The starting node.
    end : Node
        The ending node.

    Returns
    -------
    nbrs : Set
        The set of neighbors that are also along a path towards
        the 'end' node.
    """
    nbrs = set()

    # query all neighbors of X and then only add nodes that are in a valid path
    # to end
    for node in G.neighbors(start):
        if not G.has_edge(start, node):
            raise RuntimeError(f"{start} and {node} are not connected, but they are assumed to be.")

        # if we queried the edge we are testing, then pick that one
        if node == end:
            continue

        # find a path from start node to end
        paths = nx.all_simple_paths(G, source=node, target=end)
        for path in paths:
            # the trivial path which indicates that 'node' is only connected to
            # 'end' through 'start'
            if path == (node, start, end):
                continue
            else:
                # found a single path
                nbrs.add(node)
                break
    return nbrs


class BaseSkeletonLearner:
    """Base class for constraint-based skeleton learning algorithms.

    Attributes
    ----------
    adj_graph_ : nx.Graph
        The learned skeleton graph.
    sep_set_ : SeparatingSet
        The learned separating sets.
    context_ : Context
        The resulting causal context.
    n_iters_ : int
        The number of iterations of the skeleton learning process that were performed.
        This helps track iteration of algorithms that perform the entire skeleton
        discovery phase multiple times.
    """

    alpha: float
    n_jobs: Optional[int]

    adj_graph_: nx.Graph
    context_: Context
    sep_set_: SeparatingSet
    n_iters_: int

    min_cond_set_size_: int
    max_cond_set_size_: int
    max_combinations_: int

    _cont: bool

    def _learn_skeleton(
        self,
        data: pd.DataFrame,
        context: Context,
        condsel_method: ConditioningSetSelection,
        conditional_test_func: Callable[[Column, Column, Set[Column]], Tuple[float, float]],
        possible_x_nodes=None,
        skipped_y_nodes=None,
        skipped_z_nodes=None,
        cross_distribution_test: bool = False,
        group_with_snode=None,
        debug: bool = False,
    ):
        """Core function for learning the skeleton of a causal graph.

        This function is a "stateful" function of Skeleton learners. It requires
        the ``context_`` object to be preserved as attributes of self. It also keeps
        track of ``_cont`` private attribute, which helps determine stopping conditions.

        Parameters
        ----------
        data : pd.DataFrame
            The data to learn the causal graph from.
        context : Context
            A context object.
        condsel_method : ConditioningSetSelection
            Method to compute candidate conditioning set.
        conditional_test_func : Callable
            The conditional test function that takes in three arguments 'x_var', 'y_var'
            and an optional 'z_var', where 'z_var' is the conditioning set of variables.
        possible_x_nodes : set of nodes, optional
            The nodes to initialize as X variables. How to initialize variables to test in
            the second loop of the algorithm. See Notes for details.
        skipped_y_nodes : set of nodes, optional
            The nodes to skip in choosing the Y variable. See Notes for details.
        skipped_z_nodes : set of nodes, optional
            The nodes to skip in choosing the conditioning set. See Notes for details.
        cross_distribution_test : bool, optional
            Whether to perform cross-distribution tests. If True, then the ``context``
            object must contain a ``sigma_map`` attribute that maps each X-node
            to the corresponding distributions of interest.
        cross_domain_test : bool, optional
            Whether to perform cross-domain tests. If True, then the ``context``
            object must contain a ``domain_map`` attribute that maps each X-node
            to the corresponding domains of interest.

        Notes
        -----
        The context object should be copied before this function is called.

        Proceed by testing neighboring nodes, while keeping track of test
        statistic values (these are the ones that are
        the "most dependent"). Remember we are testing the null hypothesis

        .. math::
            H_0: X \\perp Y | Z

        where the alternative hypothesis is that they are dependent and hence
        require a causal edge linking the two variables.

        Overview of learning causal skeleton from data:

            This algorithm consists of four general loops through the data.

            1. "Infinite" loop through size of the conditioning set, 'size_cond_set'. The
            minimum size is set by ``min_cond_set_size``, whereas the maximum is controlled
            by ``max_cond_set_size`` hyperparameter.
            2. Loop through nodes of the graph, 'x_var'
            3. Loop through variables adjacent to selected node, 'y_var'. The edge between 'x_var'
            and 'y_var' is tested with a statistical test.
            4. Loop through combinations of the conditioning set of size p, 'cond_set'.
            The ``max_combinations`` parameter allows one to limit the fourth loop through
            combinations of the conditioning set.

            At each iteration of the outer infinite loop, the edges that were deemed
            independent for a specific 'size_cond_set' are removed and 'size_cond_set'
            is incremented.

            Furthermore, the maximum pvalue is stored for existing
            dependencies among variables (i.e. any two nodes with an edge still).
            The ``keep_sorted`` hyperparameter keeps the considered neighbors in
            a sorted order.

            The stopping condition is when the size of the conditioning variables for all (X, Y)
            pairs is less than the size of 'size_cond_set', or if the 'max_cond_set_size' is
            reached.
        """
        # get the initialized graph and do not make a copy, since we will be modifying it
        # in place in the context object
        if possible_x_nodes is None:
            possible_x_nodes = context.init_graph.nodes
        if skipped_y_nodes is None:
            skipped_y_nodes = set()
        if skipped_z_nodes is None:
            skipped_z_nodes = set()

        # the size of the conditioning set will start off at the minimum
        size_cond_set = self.min_cond_set_size_

        logger.info(
            f"\n\nRunning skeleton phase with: \n"
            f"max_combinations: {self.max_combinations_},\n"
            f"min_cond_set_size: {self.min_cond_set_size_},\n"
            f"max_cond_set_size: {self.max_cond_set_size_},\n"
        )

        # Outer loop: iterate over 'size_cond_set' until stopping criterion is met
        # - 'size_cond_set' > 'max_cond_set_size' or
        # - All (X, Y) pairs have candidate conditioning sets of size < 'size_cond_set'
        while 1:
            # private attribute '_cont' is used to preserve state and determine a breaking
            # condition for the constraint-based search algorithm
            self._cont = False

            # initialize set of edges to remove at the end of every loop
            # track progress of the algorithm for which edges to remove to ensure stability
            # wrt which edges are removed at each process of the algorithm
            remove_edges = set()

            if self.n_jobs == 1 or self.n_jobs is None:
                for x_var, y_var, possible_variables in self._generate_pairs_with_sepset(
                    possible_x_nodes,
                    context,
                    condsel_method,
                    size_cond_set,
                    skipped_y_nodes,
                    skipped_z_nodes,
                ):
                    # generate iterator through the conditioning sets
                    conditioning_sets = _iter_conditioning_set(
                        possible_variables=possible_variables,
                        x_var=x_var,
                        y_var=y_var,
                        size_cond_set=size_cond_set,
                    )

                    # now iterate through the possible parents
                    for comb_idx, cond_set in enumerate(conditioning_sets):
                        # check the number of combinations of possible parents we have tried
                        # to use as a separating set
                        if (
                            self.max_combinations_ is not None
                            and comb_idx >= self.max_combinations_
                        ):
                            break

                        # either process within-distribution, or across distributions
                        this_data = data
                        if cross_distribution_test:
                            # compute conditional independence test
                            # get the sigma-map for this F-node
                            distribution_idx = context.sigma_map[x_var]

                            # get the distributions across the two distributions
                            data_i = data[distribution_idx[0]].copy()
                            data_j = data[distribution_idx[1]].copy()

                            # name the group column the F-node, so Oracle works as expected
                            data_i[x_var] = 0
                            data_j[x_var] = 1
                            this_data = pd.concat((data_i, data_j), axis=0)

                            if group_with_snode is not None:
                                old_xvar = x_var
                                x_var = {x_var, group_with_snode}
                        try:
                            test_stat, pvalue = self.evaluate_edge(
                                this_data, conditional_test_func, x_var, y_var, set(cond_set)
                            )
                        except Exception as e:
                            # allow us to catch exceptions that are due to not enough samples
                            # if so, we cannot remove the edge and just proceed
                            test_stat = np.inf
                            pvalue = 0.0

                        if debug:
                            print(f'Comparing {x_var}-{y_var} | {cond_set} : {pvalue}')

                        # if any "independence" is found through inability to reject
                        # the null hypothesis, then we will break the loop comparing X and Y
                        # and say X and Y are conditionally independent given 'cond_set'
                        if pvalue > self.alpha:
                            break

                    # post-process the CI test results
                    if group_with_snode is not None:
                        self._postprocess_ci_test(context, group_with_snode, y_var, test_stat, pvalue)
                        self._postprocess_ci_test(context, old_xvar, y_var, test_stat, pvalue)
                    else:
                        self._postprocess_ci_test(context, x_var, y_var, test_stat, pvalue)

                    # two variables found to be independent given a separating set
                    if pvalue > self.alpha:
                        self.sep_set_[x_var][y_var].append(set(cond_set))
                        self.sep_set_[y_var][x_var].append(set(cond_set))
                        remove_edges.add((x_var, y_var, pvalue))

                    # summarize the comparison of XY
                    self._summarize_xy_comparison(x_var, y_var, pvalue > self.alpha, pvalue)
            else:
                # run parallelized loop
                out = Parallel(n_jobs=self.n_jobs)(
                    delayed(_parallel_test_xy_edges)(
                        self.evaluate_edge,
                        conditional_test_func,
                        x_var,
                        y_var,
                        self.alpha,
                        size_cond_set,
                        self.max_combinations_,
                        possible_variables,
                        data,
                        context,
                    )
                    for x_var, y_var, possible_variables in self._generate_pairs_with_sepset(
                        possible_x_nodes,
                        context,
                        condsel_method,
                        size_cond_set,
                        skipped_y_nodes,
                        skipped_z_nodes,
                    )
                )

                for result in out:
                    test_stat = result["test_stat"]
                    pvalue = result["pvalue"]
                    x_var = result["x_var"]
                    y_var = result["y_var"]
                    cond_set = result["cond_set"]

                    # post-process the CI test results
                    self._postprocess_ci_test(context, x_var, y_var, test_stat, pvalue)

                    # two variables found to be independent given a separating set
                    if pvalue > self.alpha:
                        self.sep_set_[x_var][y_var].append(set(cond_set))
                        self.sep_set_[y_var][x_var].append(set(cond_set))
                        remove_edges.add((x_var, y_var, pvalue))

                    # summarize the comparison of XY
                    self._summarize_xy_comparison(x_var, y_var, pvalue > self.alpha, pvalue)

            # finally remove edges after performing
            # conditional independence tests
            logger.info(f"For p = {size_cond_set}, removing all edges: {remove_edges}")

            # Remove non-significant links
            # Note: Removing edges at the end ensures "stability" of the algorithm
            # with respect to the randomness choice of pairs of edges considered in the inner loop
            context.init_graph.remove_edges_from(remove_edges)

            # increment the conditioning set size
            size_cond_set += 1

            # only allow conditioning set sizes up to maximum set number
            if size_cond_set > self.max_cond_set_size_ or self._cont is False:
                break

        self.adj_graph_ = context.init_graph
        self.n_iters_ += 1

    def _generate_pairs_with_sepset(
        self,
        possible_x_nodes: Set[Column],
        context: Context,
        condsel_method: ConditioningSetSelection,
        size_cond_set: int,
        skipped_y_nodes: Set[Column],
        skipped_z_nodes: Set[Column],
    ) -> Generator[Tuple[Column, Column, Set[Column]], None, None]:
        """Generate X, Y and Z pairs for conditional testing.

        Parameters
        ----------
        possible_x_nodes : Set[Column]
            Nodes that we want to test edges of.
        context : Context
            The causal context. Must containt the graph encoding adjacencies and current
            state of the learned undirected graph.
        condsel_method : ConditioningSetSelection
            The method to use for selecting conditioning sets.
        size_cond_set : int
            The current size of the conditioning set to consider.
        skipped_y_nodes : Set[Column]
            Allow one to skip Y-nodes that are not of interest in learning edge structure.
        skipped_z_nodes : Set[Column]
            Allow one to skip Z-nodes that are not able to be conditioned on.

        Yields
        ------
        Generator[Tuple[Column, Column, Set[Column]], None, None]
            Generates 'X' variable, 'Y' variable and canddiate 'Z' (i.e. possible separating set
            variables).
        """
        # keep a memoization of (x_var, y_var) pairs we have already seen
        seen_pairs = set()

        # loop through every node that we want to test
        for x_var in possible_x_nodes:
            possible_adjacencies = set(context.init_graph.neighbors(x_var))
            logger.info(f"Considering node {x_var}...\n\n")

            for y_var in possible_adjacencies:
                # ignore any nodes we do not want to consider's X-Y edges of
                if y_var in skipped_y_nodes:
                    continue

                # prevent yielding the same edge pair twice
                if (x_var, y_var) in seen_pairs or (y_var, x_var) in seen_pairs:
                    continue

                # a node cannot be a parent to itself in DAGs
                if y_var == x_var:
                    continue

                # leverage prior knowledge to skip edges if they are included
                if (x_var, y_var) in context.included_edges.edges:
                    continue

                # compute the possible variables used in the conditioning set
                possible_variables = candidate_cond_sets(
                    condsel_method, context, x_var, y_var, keep_sorted=self.keep_sorted
                )

                # remove nodes that are not allowed to be conditioned on
                # XXX: if used, this may result in improper graphs learned even in oracle setting
                possible_variables = possible_variables.difference(skipped_z_nodes)

                logger.debug(
                    f"Adj({x_var}) without {y_var} with size={len(possible_adjacencies) - 1} "
                    f"with p={size_cond_set}. The possible variables to condition on are: "
                    f"{possible_variables}."
                )

                # check that number of adjacencies is greater then the
                # cardinality of the conditioning set
                if len(possible_variables) < size_cond_set:
                    logger.debug(
                        f"\n\nBreaking for {x_var}, {y_var}, {len(possible_adjacencies)}, "
                        f"{size_cond_set}, {possible_variables}"
                    )
                    continue
                else:
                    self._cont = True

                seen_pairs.add((x_var, y_var))
                yield x_var, y_var, possible_variables

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column
    ) -> Set[Column]:
        r"""Compute candidate conditioning sets.

        For a given 'X' and 'Y', this method implements a graphical algorithm that
        enumerates possible variables that are part of 'Z', the conditioning set.
        One can then test the following null hypothesis :math:`H_0: X \perp Y | Z`.

        Parameters
        ----------
        adj_graph : nx.Graph
            The current adjacency graph.
        x_var : node
            The 'X' node.
        y_var : node
            The 'Y' node.

        Returns
        -------
        possible_variables : Set of Column
            The set of nodes in 'adj_graph' that are candidates for the
            conditioning set.

        Notes
        -----
        This depends on:

        size_cond_set : int
            The maximum size of the conditioning set allowed. If candidate conditioning
            sets are less than this number, then the ``possible_variables`` will be
            the empty set.
        """
        raise NotImplementedError(
            "All skeleton discovery methods should implement a method for selecting "
            "the possible conditioning sets."
        )

    def _postprocess_ci_test(
        self,
        adj_graph: nx.Graph,
        x_var: Column,
        y_var: Column,
        test_stat: float,
        pvalue: float,
    ):
        """Post-processing of CI tests.

        The basic values any learner keeps track of is the pvalue/test-statistic of each
        remaining edge. This is a heuristic estimate of the "dependency" of any node
        with its neighbors.

        Parameters
        ----------
        adj_graph : nx.Graph
            The adjacency graph.
        x_var : Column
            X variable.
        y_var : Column
            Y variable.
        test_stat : float
            The test statistic.
        pvalue : float
            The pvalue of the test statistic.
        """
        # keep track of the smallest test statistic, meaning the highest pvalue
        # meaning the "most" independent. keep track of the maximum pvalue as well
        if pvalue > adj_graph.edges[x_var, y_var]["pvalue"]:
            adj_graph.edges[x_var, y_var]["pvalue"] = pvalue
        if test_stat < adj_graph.edges[x_var, y_var]["test_stat"]:
            adj_graph.edges[x_var, y_var]["test_stat"] = test_stat

    def _summarize_xy_comparison(
        self, x_var: Column, y_var: Column, removed_edge: bool, pvalue: float
    ) -> None:
        """Provide ability to log end result of each XY edge evaluation."""
        # exit loop if we have found an independency and removed the edge
        if removed_edge:
            remove_edge_str = "Removing edge"
        else:
            remove_edge_str = "Did not remove edge"

        logger.info(
            f"{remove_edge_str} between {x_var} and {y_var}... \n"
            f"Statistical summary:\n"
            f"- PValue={pvalue} at alpha={self.alpha}"
        )

    def evaluate_edge(
        self, data: pd.DataFrame, X: Column, Y: Column, Z: Optional[Set[Column]] = None
    ) -> Tuple[float, float]:
        """Test any specific edge for X || Y | Z.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset
        X : column
            A column in ``data``.
        Y : column
            A column in ``data``.
        Z : set, optional
            A list of columns in ``data``, by default None.

        Returns
        -------
        test_stat : float
            Test statistic.
        pvalue : float
            The pvalue.
        """
        raise NotImplementedError(
            "All skeleton discovery methods should implement a method for "
            "evaluating an edge with a statistical test."
        )


class LearnSkeleton(BaseSkeletonLearner):
    """Learn a skeleton graph from observational data without latent confounding.

    A skeleton graph from a Markovian causal model can be learned completely
    with this procedure.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function.
    sep_set : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
        will be initialized.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        The maximum number of conditional independence tests to run from the set
        of possible conditioning sets. By default None, which means the algorithm will
        check all possible conditioning sets. If ``max_combinations=n`` is set, then
        for every conditioning set size, 'p', there will be at most 'n' CI tests run
        before the conditioning set size 'p' is incremented. For controlling the size
        of 'p', see ``min_cond_set_size`` and ``max_cond_set_size``. This can be used
        in conjunction with ``keep_sorted`` parameter to only test the "strongest"
        dependences.
    condsel_method : ConditioningSetSelection
        The method to use for selecting the conditioning set. Must be one of
        ('complete', 'neighbors', 'neighbors_path'). See Notes for more details.
    keep_sorted : bool
        Whether or not to keep the considered conditioning set variables in sorted
        dependency order. If True (default) will sort the existing dependencies of each variable
        by its dependencies from strongest to weakest (i.e. largest CI test statistic value
        to lowest). This can be used in conjunction with ``max_combinations`` parameter
        to only test the "strongest" dependences.
    n_jobs : int, optional
        Number of CPUs to use, by default None.

    Attributes
    ----------
    adj_graph_ : nx.Graph
        The discovered graph from data. Stored using an undirected
        graph. The graph contains edge attributes for the smallest value of the
        test statistic encountered (key name 'test_stat'), the largest pvalue seen in
        testing 'x' || 'y' given some conditioning set (key name 'pvalue').
    sep_set_ : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
    context_ : Context
        The result context. Encodes causal assumptions.
    min_cond_set_size_ : int
        The inferred minimum conditioning set size.
    max_cond_set_size_ : int
        The inferred maximum conditioning set size.
    max_combinations_ : int
        The inferred maximum number of combinations of 'Z' to test per
        :math:`X \\perp Y | Z`.
    n_iters_ : int
        The number of iterations the skeleton has been learned.

    Notes
    -----
    Proceed by testing neighboring nodes, while keeping track of test
    statistic values (these are the ones that are
    the "most dependent"). Remember we are testing the null hypothesis

    .. math::
        H_0: X \\perp Y | Z

    where the alternative hypothesis is that they are dependent and hence
    require a causal edge linking the two variables.

    Different methods for learning the skeleton:

        There are different ways to learn the skeleton that are valid under various
        assumptions. The value of ``condsel_method`` completely defines how one
        selects the conditioning set.

        - 'complete': This exhaustively conditions on all combinations of variables in
          the graph. This essentially refers to the SGS algorithm :footcite:`Spirtes1993`
        - 'neighbors': This only conditions on adjacent variables to that of 'x_var' and 'y_var'.
          This refers to the traditional PC algorithm :footcite:`Meek1995`
        - 'neighbors_path': This is 'neighbors', but restricts to variables with an adjacency path
          from 'x_var' to 'y_var'. This is a variant from the RFCI paper :footcite:`Colombo2012`
    """

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        sep_set: Optional[SeparatingSet] = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        condsel_method: ConditioningSetSelection = ConditioningSetSelection.NBRS,
        keep_sorted: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.ci_estimator = ci_estimator
        self.sep_set = sep_set
        self.alpha = alpha
        self.condsel_method = condsel_method
        self.n_jobs = n_jobs

        # control of the conditioning set
        self.min_cond_set_size = min_cond_set_size
        self.max_cond_set_size = max_cond_set_size
        self.max_combinations = max_combinations

        # for tracking strength of dependencies
        self.keep_sorted = keep_sorted

        # debugging mode
        self.n_ci_tests = 0
        self.n_iters_ = 0

    def _initialize_params(self) -> None:
        """Initialize parameters for learning skeleton.

        Basic parameters that are used by any constraint-based causal discovery algorithms.
        """
        # error checks of passed in arguments
        if self.max_combinations is not None and self.max_combinations <= 0:
            raise RuntimeError(f"Max combinations must be at least 1, not {self.max_combinations}")

        if self.condsel_method not in ConditioningSetSelection:
            raise ValueError(
                f"Skeleton method must be one of {ConditioningSetSelection}, not "
                f"{self.condsel_method}."
            )

        if self.sep_set is None and not hasattr(self, "sep_set_"):
            # keep track of separating sets
            self.sep_set_ = defaultdict(lambda: defaultdict(list))
        elif not hasattr(self, "sep_set_"):
            self.sep_set_ = self.sep_set  # type: ignore

        # control of the conditioning set
        if self.max_cond_set_size is None:
            self.max_cond_set_size_ = np.inf
        else:
            self.max_cond_set_size_ = self.max_cond_set_size
        if self.min_cond_set_size is None:
            self.min_cond_set_size_ = 0
        else:
            self.min_cond_set_size_ = self.min_cond_set_size
        if self.max_combinations is None:
            self.max_combinations_ = np.inf
        else:
            self.max_combinations_ = self.max_combinations

    def evaluate_edge(
        self, data: pd.DataFrame, X: Column, Y: Column, Z: Optional[Set[Column]] = None
    ) -> Tuple[float, float]:
        """Test any specific edge for X || Y | Z.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset
        X : column
            A column in ``data``.
        Y : column
            A column in ``data``.
        Z : set, optional
            A list of columns in ``data``, by default None.

        Returns
        -------
        test_stat : float
            Test statistic.
        pvalue : float
            The pvalue.
        """
        if Z is None:
            Z = set()
        test_stat, pvalue = self.ci_estimator.test(data, set({X}), set({Y}), Z)
        self.n_ci_tests += 1
        return test_stat, pvalue

    def fit(self, data: pd.DataFrame, context: Context):
        """Run structure learning to learn the skeleton of the causal graph.

        Parameters
        ----------
        data : pd.DataFrame
            The data to learn the causal graph from.
        context : Context
            A context object.
        """
        self.context_ = context.copy()

        # initialize learning parameters
        self._initialize_params()

        # allow us to query the iteration stage of the causal discovery algorithm
        # allowing us to run multiple iterations of the skeleton discovery
        edge_attrs = set(
            chain.from_iterable(d.keys() for *_, d in self.context_.init_graph.edges(data=True))
        )
        if self.n_iters_ == 0 and "test_stat" in edge_attrs or "pvalue" in edge_attrs:
            raise RuntimeError(
                "Running skeleton discovery with adjacency graph "
                "with 'test_stat' or 'pvalue' is not supported yet."
            )

        # store the absolute value of test-statistic values and pvalue for
        # every single candidate parent-child edge (X -> Y)
        nx.set_edge_attributes(self.context_.init_graph, np.inf, "test_stat")
        nx.set_edge_attributes(self.context_.init_graph, -1e-5, "pvalue")

        # apply algorithm to learn skeleton
        self._learn_skeleton(
            data,
            context=self.context_,
            conditional_test_func=self.evaluate_edge,
        )

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column
    ) -> Set[Column]:
        r"""Compute candidate conditioning sets.

        For a given 'X' and 'Y', this method implements a graphical algorithm that
        enumerates possible variables that are part of 'Z', the conditioning set.
        One can then test the following null hypothesis :math:`H_0: X \perp Y | Z`.

        Parameters
        ----------
        adj_graph : nx.Graph
            The current adjacency graph.
        x_var : node
            The 'X' node.
        y_var : node
            The 'Y' node.

        Returns
        -------
        possible_variables : Set of Column
            The set of nodes in 'adj_graph' that are candidates for the
            conditioning set.

        Notes
        -----
        The :attr:`condsel_method` dictates how we choose the corresponding conditioning sets.
        For more information, see :class:`ConditioningSetSelection`.
        """
        condsel_method = self.condsel_method

        if condsel_method == ConditioningSetSelection.COMPLETE:
            possible_variables = set(adj_graph.nodes)
        elif condsel_method == ConditioningSetSelection.NBRS:
            possible_variables = set(adj_graph.neighbors(x_var))
            # possible_adjacencies.copy()
        elif condsel_method == ConditioningSetSelection.NBRS_PATH:
            # constrain adjacency set to ones with a path from x_var to y_var
            possible_variables = _find_neighbors_along_path(adj_graph, start=x_var, end=y_var)

        if self.keep_sorted:
            # Note it is assumed in public API that 'test_stat' is set
            # inside the adj_graph
            possible_variables = sorted(
                possible_variables,
                key=lambda n: adj_graph.edges[x_var, n]["test_stat"],
                reverse=True,
            )  # type: ignore

        if x_var in possible_variables:
            possible_variables.remove(x_var)
        if y_var in possible_variables:
            possible_variables.remove(y_var)

        return possible_variables


class LearnSemiMarkovianSkeleton(LearnSkeleton):
    """Learning a skeleton from a semi-markovian causal model.

    This proceeds by learning a skeleton by testing edges with candidate
    separating sets from the "possibly d-separating" sets (PDS), or PDS
    sets that lie on a path between two nodes :footcite:`Spirtes1993`.
    This algorithm requires the input of a collider-oriented PAG, which
    provides the necessary information to compute the PDS set for any
    given nodes. See Notes for more details.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function.
    sep_set : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
        will be initialized.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        The maximum number of conditional independence tests to run from the set
        of possible conditioning sets. By default None, which means the algorithm will
        check all possible conditioning sets. If ``max_combinations=n`` is set, then
        for every conditioning set size, 'p', there will be at most 'n' CI tests run
        before the conditioning set size 'p' is incremented. For controlling the size
        of 'p', see ``min_cond_set_size`` and ``max_cond_set_size``. This can be used
        in conjunction with ``keep_sorted`` parameter to only test the "strongest"
        dependences.
    condsel_method : ConditioningSetSelection
        The method to use for determining conditioning sets when testing conditional
        independence of the first stage. See :class:`LearnSkeleton` for details.
    second_stage_condsel_method : ConditioningSetSelection | None
        The method to use for determining conditioning sets when testing conditional
        independence of the first stage. Must be one of ('pds', 'pds_path'). See Notes
        for more details. If `None`, then no second stage skeleton discovery phase will be run.
    keep_sorted : bool
        Whether or not to keep the considered conditioning set variables in sorted
        dependency order. If True (default) will sort the existing dependencies of each variable
        by its dependencies from strongest to weakest (i.e. largest CI test statistic value
        to lowest). This can be used in conjunction with ``max_combinations`` parameter
        to only test the "strongest" dependences.
    max_path_length : int, optional
        The maximum length of any discriminating path, or None if unlimited.

    Attributes
    ----------
    adj_graph_ : nx.Graph
        The discovered graph from data. Stored using an undirected
        graph. The graph contains edge attributes for the smallest value of the
        test statistic encountered (key name 'test_stat'), the largest pvalue seen in
        testing 'x' || 'y' given some conditioning set (key name 'pvalue').
    sep_set_ : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
    context_ : Context
        The result context. Encodes causal assumptions.
    min_cond_set_size_ : int
        The inferred minimum conditioning set size.
    max_cond_set_size_ : int
        The inferred maximum conditioning set size.
    max_combinations_ : int
        The inferred maximum number of combinations of 'Z' to test per
        :math:`X \\perp Y | Z`.
    n_iters_ : int
        The number of iterations the skeleton has been learned.
    max_path_length_ : int
        Th inferred maximum path length any single discriminating path is allowed to take.
    n_jobs : int, optional
        Number of CPUs to use, by default None.

    Notes
    -----
    To learn the skeleton of a Semi-Markovian causal model, one approach is to consider
    the possibly d-separating (PDS) set, which is a superset of the d-separating sets in
    the true causal model. Knowing the PDS set requires knowledge of the skeleton and orientation
    of certain edges. Therefore, we first learn an initial skeleton by checking conditional
    independences with respect to node neighbors. From this, one can orient certain colliders.
    The resulting PAG can now be used to enumerate the PDS sets for each node, which
    are now conditioning candidates to check for conditional independence.

    For visual examples, see Figures 16, 17 and 18 in :footcite:`Spirtes1993`. Also,
    see the RFCI paper for other examples :footcite:`Colombo2012`.

    Different methods for learning the skeleton:

        There are different ways to learn the skeleton that are valid under various
        assumptions. The value of ``condsel_method`` completely defines how one
        selects the conditioning set.

        - 'pds': This conditions on the PDS set of 'x_var'. Note, this definition does
          not rely on 'y_var'. See :footcite:`Spirtes1993`.
        - 'pds_path': This is 'pds', but restricts to variables with a possibly directed path
          from 'x_var' to 'y_var'. This is a variant from the RFCI paper :footcite:`Colombo2012`.

    References
    ----------
    .. footbibliography::
    """

    max_path_length_: int

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        sep_set: Optional[SeparatingSet] = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        condsel_method: ConditioningSetSelection = ConditioningSetSelection.NBRS,
        second_stage_condsel_method: Optional[
            ConditioningSetSelection
        ] = ConditioningSetSelection.PDS,
        keep_sorted: bool = False,
        max_path_length: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__(
            ci_estimator,
            sep_set,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            condsel_method,
            keep_sorted,
            n_jobs=n_jobs,
        )

        self.second_stage_condsel_method = second_stage_condsel_method
        self.max_path_length = max_path_length

    def _orient_unshielded_triples(self, graph: EquivalenceClass, sep_set: SeparatingSet) -> None:
        """Orient colliders given a graph and separation set.

        Parameters
        ----------
        graph : EquivalenceClass
            The partial ancestral graph (PAG).
        sep_set : SeparatingSet
            The separating set between any two nodes.
        """
        # for every node in the PAG, evaluate neighbors that have any edge
        for u in graph.nodes:
            for v_i, v_j in combinations(graph.neighbors(u), 2):
                # Check that there is no edge of any type between
                # v_i and v_j, else this is a "shielded" collider.
                # Then check to see if 'u' is in the separating
                # set. If it is not, then there is a collider.
                if v_j not in graph.neighbors(v_i) and not is_in_sep_set(
                    u, sep_set, v_i, v_j, mode="any"
                ):
                    if graph.has_edge(v_i, u, graph.circle_edge_name):
                        graph.orient_uncertain_edge(v_i, u)
                    if graph.has_edge(v_j, u, graph.circle_edge_name):
                        graph.orient_uncertain_edge(v_j, u)

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column
    ) -> Set[Column]:
        import pywhy_graphs as pgraph

        # get PAG from the context object
        pag = self.context_.state_variable("PAG", on_missing="ignore")

        if pag is None:
            # if PAG has not been set as a state variable, then we are learning a skeleton
            # without PDS information. I.e. the normal LearnSkeleton algorithm
            return super()._compute_candidate_conditioning_sets(adj_graph, x_var, y_var)
        else:
            if not all(x in pag.nodes for x in [x_var, y_var]):
                raise RuntimeError("wtf..")
            condsel_method = self.second_stage_condsel_method

            if condsel_method == ConditioningSetSelection.PDS:
                # determine how we want to construct the candidates for separating nodes
                # perform conditioning independence testing on all combinations
                possible_variables = pgraph.pds(
                    pag, x_var, y_var, max_path_length=self.max_path_length_  # type: ignore
                )
            elif condsel_method == ConditioningSetSelection.PDS_PATH:
                # determine how we want to construct the candidates for separating nodes
                # perform conditioning independence testing on all combinations
                possible_variables = pgraph.pds_path(
                    pag, x_var, y_var, max_path_length=self.max_path_length_  # type: ignore
                )

            if self.keep_sorted:
                # Note it is assumed in public API that 'test_stat' is set
                # inside the adj_graph
                possible_variables = sorted(
                    possible_variables,
                    key=lambda n: adj_graph.edges[x_var, n]["test_stat"],
                    reverse=True,
                )  # type: ignore

        if x_var in possible_variables:
            possible_variables.remove(x_var)
        if y_var in possible_variables:
            possible_variables.remove(y_var)

        return possible_variables

    def _prep_second_stage_skeleton(self) -> Context:
        import pywhy_graphs as pgraphs

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        sep_set = self.sep_set_
        skel_graph = self.adj_graph_
        pag = pgraphs.PAG(incoming_circle_edges=skel_graph, name="PAG derived with FCI")

        # orient colliders
        self._orient_unshielded_triples(pag, sep_set)

        # convert the adjacency graph
        new_init_graph = pag.to_undirected()

        # Update the Context:
        # add the corresponding intermediate PAG now to the context
        # new initialization graph
        for (_, _, d) in new_init_graph.edges(data=True):
            if "test_stat" in d:
                d.pop("test_stat")
            if "pvalue" in d:
                d.pop("pvalue")
        context = (
            make_context(self.context_)
            .init_graph(new_init_graph)
            .state_variable("PAG", pag)
            .build()
        )
        return context

    def _initialize_params(self) -> None:
        if self.max_path_length is None:
            self.max_path_length_ = np.inf
        else:
            self.max_path_length_ = self.max_path_length

        return super()._initialize_params()

    def fit(self, data: pd.DataFrame, context: Context):
        # initially learn the skeleton without using PDS information
        super().fit(data, context)

        # if there is no second stage skeleton method to be run, then we
        # will stop with the skeleton here
        if self.second_stage_condsel_method is None:
            return self

        # setup context for the second round-of learning
        context = self._prep_second_stage_skeleton()

        # now compute all possibly d-separating sets and learn a better skeleton
        super().fit(data, context)
        return self


class LearnInterventionSkeleton(LearnSemiMarkovianSkeleton):
    """Learn skeleton using observational and interventional data.

    Parameters
    ----------
    ci_estimator : BaseConditionalIndependenceTest
        The conditional independence test function.
    cd_estimator : BaseConditionalDiscrepancyTest
        The conditional discrepancy test function.
    sep_set : dictionary of dictionary of list of set
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
        will be initialized.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        The maximum number of conditional independence tests to run from the set
        of possible conditioning sets. By default None, which means the algorithm will
        check all possible conditioning sets. If ``max_combinations=n`` is set, then
        for every conditioning set size, 'p', there will be at most 'n' CI tests run
        before the conditioning set size 'p' is incremented. For controlling the size
        of 'p', see ``min_cond_set_size`` and ``max_cond_set_size``. This can be used
        in conjunction with ``keep_sorted`` parameter to only test the "strongest"
        dependences.
    condsel_method : ConditioningSetSelection
        The method to use for testing conditional independence. Must be one of
        ('pds', 'pds_path'). See Notes for more details.
    keep_sorted : bool
        Whether or not to keep the considered conditioning set variables in sorted
        dependency order. If True (default) will sort the existing dependencies of each variable
        by its dependencies from strongest to weakest (i.e. largest CI test statistic value
        to lowest). This can be used in conjunction with ``max_combinations`` parameter
        to only test the "strongest" dependences.
    max_path_length : int, optional
        The maximum length of any discriminating path, or None if unlimited.
    n_jobs : int, optional
        Number of CPUs to use, by default None.

    Notes
    -----
    With interventional data, one may either know the interventional targets from each
    experimental distribution dataset, or one may not know the explicit targets. If the
    interventional targets are known, then the skeleton discovery algorithm of
    :footcite:`Kocaoglu2019characterization` is used. That is we learn the skeleton of a
    AugmentedPAG. Otherwise, we will not know the intervention targets, and use the skeleton discovery
    algorithm described in :footcite:`Jaber2020causal`. To define intervention targets, one
    must use the :class:`dodiscover.InterventionalContextBuilder`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        cd_estimator: BaseConditionalDiscrepancyTest,
        sep_set: Optional[SeparatingSet] = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        condsel_method: ConditioningSetSelection = ConditioningSetSelection.NBRS,
        second_stage_condsel_method: ConditioningSetSelection = ConditioningSetSelection.PDS,
        keep_sorted: bool = False,
        max_path_length: Optional[int] = None,
        known_intervention_targets: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__(
            ci_estimator,
            sep_set,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            condsel_method,
            second_stage_condsel_method,
            keep_sorted,
            max_path_length,
            n_jobs=n_jobs,
        )

        self.cd_estimator = cd_estimator
        self.known_intervention_targets = known_intervention_targets

    def evaluate_edge(
        self, data: pd.DataFrame, X: Column, Y: Column, Z: Optional[Set[Column]] = None
    ) -> Tuple[float, float]:
        """Test any specific edge for X || Y | Z.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset
        X : column
            A column in ``data``.
        Y : column
            A column in ``data``.
        Z : set, optional
            A list of columns in ``data``, by default None.

        Returns
        -------
        test_stat : float
            Test statistic.
        pvalue : float
            The pvalue.
        """
        if Z is None:
            Z = set()
        test_stat, pvalue = self.ci_estimator.test(data, set({X}), set({Y}), Z)
        self.n_ci_tests += 1
        return test_stat, pvalue

    def evaluate_fnode_edge(
        self,
        data: List[pd.DataFrame],
        X: Column,
        Y: Column,
        Z: Set[Column],
    ) -> Tuple[float, float]:
        """Test an edge from an F-node to a regular node for X || Y | Z.

        Tests the conditional invariance: :math:`P_{X=x}(Y | Z) = P_{X=x'}(Y|Z)`.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset
        X : column
            A column in ``data``. This is assumed to be the F-node.
        Y : column
            A column in ``data``.
        Z : set
            A list of columns in ``data``. Can be the empty set.

        Returns
        -------
        test_stat : float
            Test statistic.
        pvalue : float
            The pvalue.
        """
        # get the sigma-map for this F-node
        distribution_idx = self.context_.sigma_map[X]

        # get the distributions across the two distributions
        data_i = data[distribution_idx[0]].copy()
        data_j = data[distribution_idx[1]].copy()

        # name the group column the F-node, so Oracle works as expected
        data_i[X] = 0
        data_j[X] = 1
        data = pd.concat((data_i, data_j), axis=0)

        # compare conditional distributions P(Y | X) vs P'(Y | X), where 'group_col'
        # indicates which distribution data came from
        # test graphically if Y is d-separated from F-node given Z
        # or test statistically (Y || F-node | Z), or P(Y|Z) =? P'(Y|Z)
        test_stat, pvalue = self.cd_estimator.test(data, set({Y}), set({X}), Z)

        self.n_ci_tests += 1
        return test_stat, pvalue

    def _compute_candidate_conditioning_sets(
        self, adj_graph: nx.Graph, x_var: Column, y_var: Column
    ) -> Set[Column]:
        """Override the computation for conditioning sets.

        Parameters
        ----------
        adj_graph : nx.Graph
            _description_
        x_var : Column
            _description_
        y_var : Column
            _description_

        Returns
        -------
        Z : Set[Column]
            _description_
        """
        f_nodes = self.context_.f_nodes

        # if F-nodes is not defined, then we are simply doing learning in the observational setting
        if len(f_nodes) == 0:
            # compute the possible variables used in the conditioning set
            return super()._compute_candidate_conditioning_sets(adj_graph, x_var, y_var)
        else:
            if y_var in f_nodes:
                raise RuntimeError("This should not be the case")

            # get candidate conditioning sets that do not include the F-nodes
            possible_variables = super()._compute_candidate_conditioning_sets(
                adj_graph, x_var, y_var
            ) - set(f_nodes)

            return possible_variables

    def _learn_skeleton_with_interventions(self, interv_data: List[pd.DataFrame], context: Context):
        state_variables = context.state_variables.copy()
        self.context_ = make_context(context, create_using=InterventionalContextBuilder).build()
        for name, var in state_variables.items():
            self.context_.add_state_variable(name, var)

        # apply algorithm to learn skeleton
        self._learn_skeleton(
            data=interv_data,
            context=self.context_,
            conditional_test_func=self.evaluate_fnode_edge,
            possible_x_nodes=list(self.context_.f_nodes),
        )

    def _learn_skeleton_with_observations(self, obs_data: pd.DataFrame, context: Context):
        # get the init graph that does not contain any F-nodes
        obs_context_bld = make_context(context, create_using=ContextBuilder)
        init_graph = deepcopy(context.init_graph)

        # get the subgraph of non-f nodes
        obs_init_graph = init_graph.subgraph(context.get_non_f_nodes())

        # now learn the observational subgraph
        obs_context_bld.init_graph(obs_init_graph)
        obs_context = obs_context_bld.build()

        # get the initialized graph
        adj_graph = deepcopy(obs_context.init_graph.copy())

        # apply algorithm to learn skeleton
        self._learn_skeleton(
            data=obs_data,
            context=obs_context,
            conditional_test_func=self.evaluate_edge,
            possible_x_nodes=list(adj_graph.nodes),
        )

    def fit(self, data: List[pd.DataFrame], context: Context) -> None:
        """Fit data and context.

        Parameters
        ----------
        data : List[pd.DataFrame]
            List of dataframes corresponding to different distributions of data.
        context : Context
            Context object.
        """
        # ensure data is a list
        if isinstance(data, pd.DataFrame):
            data = [data]

        # error-check the datasets passed in match the intervention contexts
        if len(data) != context.num_distributions:
            raise RuntimeError(
                f"The number of datasets does not match the number of interventions. "
                f"You passed in {len(data)} different datasets, whereas "
                f"there are {len(context.intervention_targets)} different interventions "
                f"specified and {context.num_distributions} distributions assumed. "
                f"It is assumed that the first dataset is observational, "
                f"while the rest are interventional."
            )

        orig_context = context.copy()
        f_nodes = context.f_nodes

        if context.obs_distribution:
            # it is fine to run the first stage of the FCI algorithm, as this will
            # not result in removing any edges among the F-nodes
            obs_data = data[0]
        else:
            # if we explicitly do not have access to the observational distribution,
            # then we should choose the experimental dataset with the most samples
            largest_data_idx = np.argmax([len(df) for df in data])
            obs_data = data[largest_data_idx]

        self.context_ = context.copy()

        # initialize learning parameters
        self._initialize_params()

        # allow us to query the iteration stage of the causal discovery algorithm
        # allowing us to run multiple iterations of the skeleton discovery
        edge_attrs = set(
            chain.from_iterable(d.keys() for *_, d in context.init_graph.edges(data=True))
        )
        if self.n_iters_ == 0 and "test_stat" in edge_attrs or "pvalue" in edge_attrs:
            raise RuntimeError(
                "Running skeleton discovery with adjacency graph "
                "with 'test_stat' or 'pvalue' is not supported yet."
            )

        # store the absolute value of test-statistic values and pvalue for
        # every single candidate parent-child edge (X -> Y)
        nx.set_edge_attributes(context.init_graph, np.inf, "test_stat")
        nx.set_edge_attributes(context.init_graph, -1e-5, "pvalue")

        # first learn the skeleton using only "observational data"
        self._learn_skeleton_with_observations(obs_data, context)

        # keep track of the observational skeleton graph
        obs_skel_graph = self.adj_graph_.copy()

        # prepare the context object for the second stage of learning
        # all separating sets are either:
        # i) augmented with all F-nodes, or
        # ii) augmented with all F-nodes except intervention index 'i'
        # R9 allows us to leverage F-nodes being not in separating sets to
        # augment all separating sets that have non-empty sets with all
        # F-nodes to keep consistency with the algorithm
        for x_var, y_vars in self.sep_set_.items():
            for y_var in y_vars:
                sep_sets: List = self.sep_set_.get(x_var).get(y_var)  # type: ignore
                if len(sep_sets) > 0:
                    for idx in range(len(sep_sets)):
                        self.sep_set_[x_var][y_var][idx].update(f_nodes)

        # index all datasets, where the first one may be observational
        non_f_nodes = context.get_non_f_nodes()

        # reset the init graph and this time learn the skeleton using
        # interventional distributions
        # create a complete subgraph of F-nodes with all other nodes
        for node in f_nodes:
            for obs_node in set(non_f_nodes):
                if node == obs_node:
                    continue
                self.adj_graph_.add_edge(node, obs_node, test_stat=np.inf, pvalue=-1e-5)

        # reset context and add observational skeleton
        self.context_ = (
            make_context(orig_context, create_using=InterventionalContextBuilder)
            .init_graph(self.adj_graph_.copy())
            .build()
        )
        self.context_.add_state_variable("obs_skel_graph", obs_skel_graph)

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        sep_set = self.sep_set_
        import pywhy_graphs

        pag = pywhy_graphs.PAG(incoming_circle_edges=obs_skel_graph, name="PAG derived with FCI")

        # orient colliders
        self._orient_unshielded_triples(pag, sep_set)

        # convert the adjacency graph into a PAG
        # Note: in order to preserve PDS sets for PAG augmented with the F-node, we simply have
        # to make it fully-connected, since at this stage, the intermediate PAG learned from FCI
        # has not done anything with the F-node edges.
        for f_node in f_nodes:
            for node in non_f_nodes:
                pag.add_edge(f_node, node, pag.directed_edge_name)
        self.context_.add_state_variable("PAG", pag)

        # now, we'll fit the data using interventional data by looping over all
        # combinations of F-nodes and their neighbors
        self._learn_skeleton_with_interventions(data, self.context_)

        for x_var, y_vars in self.sep_set_.items():
            for y_var in y_vars:
                f_node = None
                if x_var in f_nodes:
                    f_node = x_var
                if y_var in f_nodes:
                    f_node = y_var

                if f_node is None:
                    continue

                f_nodes_without_this = f_nodes.copy()
                f_nodes_without_this.remove(f_node)
                sep_sets: List = self.sep_set_.get(x_var).get(y_var)  # type: ignore
