"""
File Name: Joint_Entropy.py

Purpose: This module implements joint entropy for multiple variables
with automatic handling of continuous and categorical data.

Joint Entropy for two discrete random variables is defined as:

H(X,Y) =  -∑ p(x,y) log₂ p(x,y)

Joint Entropy for three discrete random variables is defined as:

H(X,Y,Z) = -∑ p(x,y,z) log₂ p(x,y,z)
"""
import numpy as np
import pandas as pd
from collections import Counter

from .Entropy import discretize_if_needed
#=================================================================================
# joint entropy function for any number of variables
def joint_entropy(*variables, bins = 10):
    """
    Computes joint entropy H(X1, X2, ..., Xn) for any number of variables.
    Automatically discretizes continous variables, handles categorical 
    variables, and aligns and removes missing values. 

    Returns:
        float: joint entropy in bits
    """
    # convert all inputs to Series and discretize if needed
    processed = []
    for var in variables: 
        var = np.asarray(var).ravel()
        v = discretize_if_needed(var, bins = bins)
        processed.append(pd.Series(v).reset_index(drop = True))

    # combine into DataFrame
    df = pd.concat(processed, axis = 1)
    df = df.dropna()

    if len(df) == 0:
        return 0.0
    
    # convert rows into tuples
    joint_states = [tuple(row) for row in df.to_numpy()]

    counts = Counter(joint_states)

    probs = [count / len(joint_states) for count in counts.values()]

    return -sum(p * np.log2(p) for p in probs if p > 0)
