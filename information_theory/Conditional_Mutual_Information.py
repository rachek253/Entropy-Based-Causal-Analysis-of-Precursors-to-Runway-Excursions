"""
File Name: Conditional_Mutual_Information.py

Purpose: This module implements conditional mutual information for
discrete random variables. Conditional mutual information measures
the amount of information shared between two variables after accounting
for the influence of a third variable. Conditional Mutual Information (CMI) 
tells us how much information X gives about Y, given that we already know Z.
CMI is useful to detect direct relationships while controlling for another 
variable.

Conditional Mutual Information is defined as:

I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
"""
from .Entropy import entropy
from .Joint_Entropy import joint_entropy
#=================================================================================
def conditional_mutual_information(x, y, z, bins = 10):
    """
    Computes conditional mutual information I(X;Y|Z)
    """
    return (
        joint_entropy(x, z, bins = bins) 
        + joint_entropy(y, z, bins = bins)
        - entropy(z, bins)
        - joint_entropy(x, y, z, bins = bins)
    )