"""
File Name: core_CSE.py

Purpose: This module is the core Causation Entropy module that implements
forward feature selection using CSE, permutation testing, and bootstrap 
stability. This module is considered the heart of the causal discovery
pipeline.
"""
import numpy as np
import pandas as pd

from sklearn.utils import resample

from information_theory.Mutual_Information import mutual_information
from information_theory.Conditional_Mutual_Information import conditional_mutual_information
from information_theory.Causation_Entropy import causation_entropy

from causal_analysis.utils import combine_features
#=================================================================================
# configuration for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
#=================================================================================

#=================================================================================
def forward_causation_entropy(df, target_col, candidate_features, max_features = 6):
    """
    Performs greedy forward selection using causation entropy. At each
    step, the feature that maximizes conditional mutual information 
    with the target (given the already-selected features) is added.

    Returns:
        - selected_features (list): Ordered list of selected features.
        - scores (dict): CMI score for each selected feature.
    """
    selected_features = []
    scores = {}

    remaining_features = candidate_features.copy()

    for _ in range(min(max_features, len(candidate_features))):
        best_feature = None
        best_score = -np.inf

        for feature in remaining_features:
            x = combine_features(df[[feature]])
            y = combine_features(df[[target_col]])

            if selected_features:
                z = combine_features(df[selected_features])
            else:
                z = None
            
            score = causation_entropy(x, y, z)

            if score > best_score:
                best_score = score
                best_feature = feature

        if best_feature is None:
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        scores[best_feature] = best_score

    return selected_features, scores
#=================================================================================

#=================================================================================
def permutation_test(df, feature, target_col, conditioning_set = None, n_permutations = 100):
    """
    Performs permutation testing for (conditional) mutual information.

    Returns:
        - p_value (float): Empirical p-value.
    """
    rng = np.random.RandomState(RANDOM_SEED)

    x = combine_features(df[[feature]])
    y = combine_features(df[[target_col]])

    if conditioning_set is None or len(conditioning_set) == 0:
        observed = mutual_information(x, y)
        z = None
    else:
        z = combine_features(df[conditioning_set])
        observed = conditional_mutual_information(x, y, z)

    permuted_scores = []

    for _ in range(n_permutations):
        y_perm = combine_features(pd.DataFrame(
            rng.permutation(df[target_col]),
            columns = [target_col]
        ))
        if conditioning_set is None or len(conditioning_set) == 0:
            score = mutual_information(x, y_perm)
        else:
            score = conditional_mutual_information(x, y_perm, z)

        permuted_scores.append(score)

    p_value = np.mean(permuted_scores >= observed)

    return p_value
#=================================================================================

#=================================================================================
def bootstrap_stability(df, target_col, candidate_features, n_bootstraps = 50, max_features = 6):
    """
    Evaluates the stability of feature selection using bootstrap
    resampling. 

    Returns:
        - stability_counts (dict): Frequency of selection per feature.
    """
    stability_counts = {f: 0 for f in candidate_features}

    for i in range(n_bootstraps):
        sample_df = resample(df, replace = True, random_state = RANDOM_SEED + i)

        selected, _ = forward_causation_entropy(
            sample_df,
            target_col,
            candidate_features,
            max_features
        )

        for f in selected:
            stability_counts[f] += 1

    # normalizing to proportions
    stability_counts = {k: v / n_bootstraps for k, v in stability_counts.items()}

    return stability_counts
#=================================================================================

#=================================================================================
# function to run analysis pipeline
def run_analysis(df):
    """
    Executes the full causal discovery pipeline. The module works by 
    first defining candidate features, then performs causation entropy 
    selection, runs permutation testing, and finally evaluates bootstrap
    stability.

    Returns:
        - results (dict)
    """

    target_col = "damage_binary"

    # candidate features (based on custom MI, sklearn MI, and CMI)
    candidate_features = [
        "weathercondition",
        "purposeofflight",
        "far", 
        "drct",
        "dwpf",
        "sknt",
        "operator",
        "alti",
        "tmpf",
        "make"
    ]

    # foward selection
    selected_features, scores = forward_causation_entropy(
        df, 
        target_col,
        candidate_features
    )

    # permutation testing
    p_values = {}
    conditioning_set = []

    for f in selected_features:
        p_val = permutation_test(
            df,
            f,
            target_col,
            conditioning_set = conditioning_set
        )

        p_values[f] = p_val
        conditioning_set.append(f)
    
    # bootstrap stability
    stability = bootstrap_stability(
        df,
        target_col,
        candidate_features
    )

    results = {
        "selected_features": selected_features,
        "scores": scores,
        "p_values": p_values,
        "stability": stability
    }

    return results
#=================================================================================
