"""
File Name: Causation_Entropy.py

Purpose: Implements causation entropy (CSE), which measures the direct
influence of one variable on another while conditioning on a set of 
variables.

Causation Entropy is defined as: CSE(X → Y | Z) = I(X; Y | Z)

CSE helps identify direct causal drivers by removing indirect effects.
"""
import numpy as np
import pandas as pd
from .Conditional_Mutual_Information import conditional_mutual_information
#=================================================================================
def causation_entropy(x, y, conditioning_set, bins = 10):
    """
    Computes causation entropy CSE(X → Y | Z) = I(X; Y | Z) with 
    multiple conditioning variables.

    Parameters:
        x: candidate cause
        y: target variable
        z: conditioning set (list or DataFrame columns)

    Returns:
        float
    """
    x = pd.Series(np.asarray(x).ravel()).reset_index(drop = True)
    y = pd.Series(np.asarray(y).ravel()).reset_index(drop = True)

    # if there is no conditioning, plain MI
    if conditioning_set is None or len(conditioning_set) == 0:
        return conditional_mutual_information(x, y, None, bins = bins)
    
    conditioning_set = pd.DataFrame(np.asarray(conditioning_set))
    
    # keeping only valid numeric/categorical columns
    for col in conditioning_set.columns:
        conditioning_set[col] = np.asarray(conditioning_set[col]).ravel()


    z = conditioning_set.reset_index(drop = True)

    return conditional_mutual_information(x, y, z, bins = bins)
    
