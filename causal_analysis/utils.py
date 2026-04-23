"""
File Name: utils.py

Purpose: This script creates utility functions utilized in Causal
Analysis. Helper functions are provided for target variable creation
and feature combination for conditioning sets. These functions ensure
consistency across the pipeline.
"""
import pandas as pd
#=================================================================================
# function to create target variable binary_damage
def create_target(df):
    """
    Create binary damage severity target variable to simplify outcome
    modeling and improve statistical stability.

    Returns:
        - DataFrame: where damage_binary = 1 for substantial or 
        destroyed, and = 0 for minor or none.
    """
    # creating target variable for feature selection
    df["damage_binary"] = df["aircraftdamage"].apply(
    lambda x: 1 if x in ["Substantial", "Destroyed"] else 0)

    return df
#=================================================================================

#=================================================================================
def combine_features(df_subset, max_unique = 100):
    """
    Combines multiple columns into a single categorical representation. 
    This is used for conditioning sets in CMI.

    Returns:
        - pandas.Series: Encoded combined feature
    """
    z= (df_subset.fillna("NA").astype(str).agg("_".join, axis = 1))
    
    if z.nunique() > max_unique:
        z= z.astype("category").cat.codes

    return z
#=================================================================================
