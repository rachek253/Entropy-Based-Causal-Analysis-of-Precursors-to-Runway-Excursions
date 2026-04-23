"""
File name: __init__.py

Causal Analysis Package

Purpose: This package implements an information-theoretic causal 
discovery pipeline using Causation Entropy (CSE), Conditional Mutual
Information (CMI), permutation-based significance testing, bootstrap
stability analysis, causal graph construction and visualization, as 
well as LaTex table generation for reporting. 

Import with:

from causal_analysis.utils import combine_features, create_target

from causal_analysis.core_CSE import(
    forward_causation_entropy, 
    permutation_test,
    bootstrap_stability,
    run_analysis
)

from causal_analysis.causal_graph import build_causal_graph, plot_graph

from causal_analysis.latex_tables import create_latex_tables, save_latex_tables
"""
