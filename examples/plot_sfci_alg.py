"""
.. _ex-sfci-algorithm:

=======================================
Causal discovery with multi-domain data
=======================================

In this example, we consider the S-FCI algorithm, which is a variant of the FCI algorithm
that learns from multi-domain data.

We will simulate linear Gaussian data that is generated from a augmented selection diagram.
That is a selection diagram with interventions applied in specific domains. Then
we will analyze how S-FCI can discover causal relationships from:

- multi-domain observational data
- multi-domain with observational data and interventional data in just the sources
- multi-domain with mixed observational and interventional data in different sources

.. currentmodule:: dodiscover
"""
