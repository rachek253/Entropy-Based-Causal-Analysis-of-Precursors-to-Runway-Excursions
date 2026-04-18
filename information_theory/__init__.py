"""
File Name: __init__.py

Information Theory Package

Purpose: This package is a collection of information theory functions 
to be used for discrete and continuous (discretized) variables. 
"""
from .Entropy import entropy, discretize_if_needed
from .Joint_Entropy import joint_entropy
from .Mutual_Information import mutual_information
from .Conditional_Mutual_Information import conditional_mutual_information
from .Causation_Entropy import causation_entropy