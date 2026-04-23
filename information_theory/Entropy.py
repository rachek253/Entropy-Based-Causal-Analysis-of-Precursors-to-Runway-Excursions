"""
File Name: Entropy.py

Purpose: This module implements the Shannon entropy function for both
discrete and continuous variables. Continuous variables are automatically
discretized before entropy calculation.

Entropy quantifies the uncertainty or 
unpredictability of a variable based on its probability distribution.

Entropy is defined as:

H(X) = -∑ p(x) log₂ p(x)

where:
-p(x) is the probability of state x
-entropy is measured in bits
"""
import numpy as np
import pandas as pd

from collections import Counter
#=================================================================================
def discretize_if_needed(x, bins = 10):
    """
    Discretizes numeric data into bins if needed.    
    """
    x = pd.Series(np.asarray(x).ravel())

    x = x.fillna(x.median() if pd.api.types.is_numeric_dtype(x)
                 else "UNKNOWN")
    if pd.api.types.is_numeric_dtype(x):
        # using rank-based discretization instead of bin edges
        x = x.rank(method = "average")
    
        # normalizing into bins
        x = (x / len(x) * bins).astype(int)
    else:
        x = x.astype("category").cat.codes

    return x
    
    
    
#=================================================================================

#=================================================================================
# defining Entropy function
def entropy(x_data, bins = 10):
    """
    Computes Shannon entropy for a variable. Automatically handles
    categorical variables and continous variables.

    Returns:
        float: entropy in bits
    """
    x = discretize_if_needed(x_data, bins = bins)

    # dropping missing values after discretization
    x = pd.Series(x).dropna()

    if len(x) == 0:
        return 0.0

    counts = Counter(x) # counting the number of data points in the array
    probs = [count/len(x) for count in counts.values()] # calculating the probability
    
    return -sum([p * np.log2(p) for p in probs if p>0])
