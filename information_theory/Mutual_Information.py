"""
File Name: Mutual_Information.py


Purpose: This module implements mutual information for pairs of discrete 
random variables. Mutual information measures the amount of information
shared between two variables and captures nonlinear dependencies 
between them. 

The interpretation of mutual information is that a value 
produced closer to 0 indicates less association between the variables, 
and a value farther from 0 indicates more association between the two
variables. 

Mutual Information is defined as:

I(X;Y) = H(X) + H(Y) - H(X,Y)
"""
from .Entropy import entropy, discretize_if_needed
from .Joint_Entropy import joint_entropy

# mutual information I(X;Y)
def mutual_information(x, y, bins = 10):
    """
    Computes mutual information I(X;Y)
    """
    x = discretize_if_needed(x, bins)
    y = discretize_if_needed(y, bins)

    return entropy(x) + entropy(y) - joint_entropy(x, y)