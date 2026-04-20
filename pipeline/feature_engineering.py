"""
File Name: feature_engineering.py

Purpose: Creates modeling-ready datasets, engineered aviation features,
and feature selection tools for runway excursion analysis.
"""
import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from information_theory.Mutual_Information import mutual_information
from information_theory.cmi_FeatureSelection import cmi_feature_selection
#=================================================================================
# feature group definitions
METAR_FEATURES = [
    "tmpf", "dwpf", "relh", "drct", "sknt",
    "gust", "vsby", "alti", "p01i",
    "crosswind_component", "tailwind_component"
]

OPERATIONAL_FEATURES = [
    "weathercondition", "purposeofflight",
    "highestinjurylevel", "far", "operator"
]

AIRCRAFT_FEATURES = [
    "make", "model"
]
# encoding categorical variables for feature selection
def build_encoded_dataset(df, target_col = "damage_binary"):
    """
    Builds ML-ready dataset by encoding categorical variables and
    retaining numerical and engineered aviation features. This function
    encodes categorical variables using one-hot encoding and removes 
    infinite and missing values. 

    Returns: 
        pandas.DataFrame
            Encoded dataset suitable for MI and CMI feature selection.
    """
    df_model = df.copy()

    df_model = df_model.dropna(subset = [target_col])

    categorical_cols = OPERATIONAL_FEATURES + AIRCRAFT_FEATURES

    numeric_cols = METAR_FEATURES

    categorical_cols = [c for c in categorical_cols if c in df_model.columns]
    numeric_cols = [c for c in numeric_cols if c in df_model.columns]

    df_encoded = pd.get_dummies(
        df_model[categorical_cols + numeric_cols + [target_col]],
        drop_first = True
    )

    df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
    df_encoded = df_encoded.dropna()

    return df_encoded
#=================================================================================

#=================================================================================
# mutual information ranking
def mutual_information_ranking(df, target_col, feature_group = None, bins = 10):
    """
    Computes mutual information between each feature and target. MI 
    measures nonlinear dependency between variables and is the primary
    feature selection method for this project. This function computes
    MI between each feature and target using custom information theory
    functions, and handles both categorical and continuous variables
    via discretization.

    Returns:
        DataFrame: ranked features by MI score.
    """
    if feature_group is None:
        features = [col for col in df.columns if col != target_col]
    else: 
        features = [f for f in feature_group if f in df.columns]

    results = []

    for col in features:
        try:
            x = df[col]

            # ensure clean format
            if x.dtype == "object":
                x = x.astype(str)
            mi = mutual_information(x, df[target_col], bins = bins)
            results.append((col, mi))
        except Exception:
            continue

    mi_df = pd.DataFrame(results, columns = ["feature", "mi_score"])
    mi_df = mi_df.sort_values(by = "mi_score", ascending = False)

    return mi_df
#=================================================================================

#=================================================================================
def sklearn_mi_ranking(df_encoded, target_col):
    """
    Computes mutual information using sklearn. This module is used for 
    comparison to determine accuracy of custom information theory 
    functions and worked on an encoded dataset only.
    """
    X = df_encoded.drop(columns = [target_col]).astype(float)
    y = df_encoded[target_col].astype(int)

    mi = mutual_info_classif(X, y, discrete_features = 'auto')

    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi
    }).sort_values(by = "mi_score", ascending = False)

    return mi_df
#=================================================================================

#=================================================================================
def cmi_feature_ranking(df_encoded, target_col, feature_group = None, tol = 0.01):
    """
    Performs greedy feature selection using Conditional Mutual Information 
    (CMI). This method selects the most informative feature using MI, 
    iteratively selects additional features using CMI, and removes
    redundant information between features. 

    Returns:
        - list: selected feature names in order of importance
        - list: corresponding MI/CMI features
    """
    if feature_group is None: 
        features = [col for col in df_encoded.columns if col != target_col]
    else:
        features = [f for f in feature_group if f in df_encoded.columns]

    x = df_encoded[features].copy()
    y = df_encoded[target_col].copy()

    def discretize_series(col):
        """
        Converts continuous variables into discrete bins for
        information-theoretic estimation stability.
        """
        if col.dtype.kind in "bifc":  # numeric types
            if col.nunique() > 10:
                return pd.qcut(col, q=10, duplicates="drop").astype(str)
            else:
                return col.astype(str)
        else:
            return col.astype(str)

    x = x.apply(discretize_series)
    y = y.astype(str)

    x_array = x.to_numpy()
    y_array = y.to_numpy()

    selected_idx, scores = cmi_feature_selection(
        y_array,
        x_array,
        tol = tol, 
        verbose = False
    )

    selected_features = [features[i] for i in selected_idx]

    return selected_features, scores
#=================================================================================

#=================================================================================
def get_feature_matrix(df_encoded, target_col):
    """
    Splits an encoded dataset into feature matrix (X) and target vector (y).
    This function ensures that MI and CMI feature selection operate on an
    identical and consistent feature space, preventing leakage or
    inconsistencies across different selection methods.

    Returns: 
        X: DataFrame of all predictors
        y: pandas.Series of target variable
    """
    X = df_encoded.drop(columns = [target_col])
    y = df_encoded[target_col]

    return X, y

#=================================================================================
# selecting most informative features
def select_topk_precursors(mi_df, top_k = 10):
    """
    Selects top k-most informative features of runway excursions based
    on MI ranking. This function assumes the input DataFrame is already 
    sorted in descending order of MI score. 

    Returns:
        list: List of feature names corresponding to top-k MI scores
    """
    return mi_df["feature"].head(top_k).tolist()
#=================================================================================

#=================================================================================
def compare_mi_cmi(df_encoded, target_col):
    """
    Compares MI vs CMI feature selection results.
    """

    mi_df =mutual_information_ranking(df_encoded, target_col)
    mi_top = mi_df.head(10)["feature"].tolist()

    cmi_features, _ = cmi_feature_ranking(df_encoded, target_col)
    
    overlap = list(set(mi_top).intersection(set(cmi_features)))

    return pd.DataFrame({
        "MI Top Features": pd.Series(mi_top),
        "CMI Selected Features": pd.Series(cmi_features),
        "Overlap": pd.Series(overlap)
    })
#=================================================================================
