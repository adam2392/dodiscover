"""
.. _ex-sfci-algorithm:

=======================================
Causal discovery with multi-domain data
=======================================

In this example, we consider the S-FCI algorithm, which is a variant of the FCI algorithm
that learns from multi-domain data. The S-FCI algorithm is described in [1]_.
The S-FCI algorithm seeks to learn an equivalence class of selection diagrams and
augmented selection diagrams. The selection diagram is a causal graph that represents
the causal relationships between variables over multiple domains
:footcite:`Bareinboim2016causal`. The augmented selection diagram is a selection diagram
augmented with F-nodes where interventions are applied as described in [1]_.

We will simulate linear Gaussian data that is generated from a augmented selection diagram.
That is a selection diagram with interventions applied in specific domains. Then
we will analyze how S-FCI can discover causal relationships from:

- multi-domain observational data
- multi-domain with observational data and interventional data in just the sources
- multi-domain with mixed observational and interventional data in different sources

In each setting, we will always have observational data in the target domain, otherwise
it would be impossible to learn how the target domain differs from the source domains.

.. currentmodule:: dodiscover

.. [1] Li A., Bareinboim E. (2023) Causal discovery of transportability from
    multi-domain data. Arxiv.
"""

# %%
# Import modules
import numpy as np
import scipy

from dodiscover.constraint import SFCI

# %%
# Simulate linear SCM (or SEM)
# ---------------------------
#
# To simulate data, we generate linear Gaussian data from an augmented selection diagram.
# This selection diagram has interventions applied in specific domains. The resulting
# data will allow us to analyze the performance of S-FCI in various scenarios.

# %%
# Learning from multi-domain observational data
# ---------------------------------------------
#
# The first scenario involves using multi-domain observational data to discover causal
# relationships. In this scenario, we examine the algorithm's ability to identify causal
# relationships between variables in different domains using only observational data.

# %%
# Learning from multi-domain interventional data
# ----------------------------------------------
#
# Next, we explore how S-FCI performs with multi-domain data that includes interventional
# data in just the sources. This scenario tests the algorithm's ability to learn from a
# combination of observational data in the target domain and interventional data in
# the sources.

# %%
# Learning from multi-domain with observational and interventional data
# ---------------------------------------------------------------------
#
# Lastly, we analyze the algorithm's performance with both observational and
# interventional data in the source domains. This scenario simulates a more
# realistic setting where data is collected from multiple domains with
# varying amounts of interventional data.
