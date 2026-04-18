"""
File Name: Causation_Entropy.py

Purpose: Implements causation entropy (CSE), which measures the direct
influence of one variable on another while conditioning on a set of 
variables.

Causation Entropy is defined as: CSE(X → Y | Z) = I(X; Y | Z)

CSE helps identify direct causal drivers by removing indirect effects.
"""
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
    z_combined = list(zip(*[conditioning_set[col] for col in conditioning_set.columns]))

    return conditional_mutual_information(x, y, z_combined, bins)