import numpy as np
from collections import Counter


def _to_1d_object_array(a):
    """
    Convert input to a 1D numpy array of dtype object.
    """
    a = np.asarray(a, dtype=object)
    if a.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return a


def _to_2d_object_array(a):
    """
    Convert input to a 2D numpy array of dtype object.
    """
    a = np.asarray(a, dtype=object)
    if a.ndim != 2:
        raise ValueError("Expected a 2D array.")
    return a


def _make_hashable_labels(z):
    """
    Convert a conditioning array z into a list of hashable labels.

    If z is 1D:
        returns [z0, z1, ..., zn]
    If z is 2D:
        returns [(z00, z01, ...), (z10, z11, ...), ...]
    """
    z = np.asarray(z, dtype=object)

    if z.ndim == 1:
        return list(z)

    if z.ndim == 2:
        return [tuple(row) for row in z.tolist()]

    raise ValueError("Conditioning set z must be 1D or 2D.")


def mutual_information(x, y, base=2):
    """
    Mutual information I(X;Y) for discrete variables.

    Parameters
    ----------
    x, y : 1D arrays
        Discrete target and feature.
    base : int or float
        Logarithm base. Use 2 for bits.

    Returns
    -------
    float
    """
    x = _to_1d_object_array(x)
    y = _to_1d_object_array(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    n = len(x)
    if n == 0:
        raise ValueError("Inputs must not be empty.")

    xy_counts = Counter(zip(x.tolist(), y.tolist()))
    x_counts = Counter(x.tolist())
    y_counts = Counter(y.tolist())

    log_base = np.log(base)
    mi = 0.0

    for (xi, yi), c_xy in xy_counts.items():
        p_xy = c_xy / n
        p_x = x_counts[xi] / n
        p_y = y_counts[yi] / n
        mi += p_xy * (np.log(p_xy / (p_x * p_y)) / log_base)

    return max(mi, 0.0)


def conditional_mutual_information(x, y, z, base=2):
    """
    Conditional mutual information I(X;Y|Z) for discrete variables.

    Parameters
    ----------
    x, y : 1D arrays
        Discrete target and feature.
    z : 1D or 2D array
        Conditioning variable(s).
    base : int or float
        Logarithm base. Use 2 for bits.

    Returns
    -------
    float
    """
    x = _to_1d_object_array(x)
    y = _to_1d_object_array(y)
    z_labels = _make_hashable_labels(z)

    if len(x) != len(y) or len(x) != len(z_labels):
        raise ValueError("x, y, and z must have the same number of samples.")

    n = len(x)
    if n == 0:
        raise ValueError("Inputs must not be empty.")

    xyz_counts = Counter((xi, yi, zi) for xi, yi, zi in zip(x.tolist(), y.tolist(), z_labels))
    xz_counts = Counter((xi, zi) for xi, zi in zip(x.tolist(), z_labels))
    yz_counts = Counter((yi, zi) for yi, zi in zip(y.tolist(), z_labels))
    z_counts = Counter(z_labels)

    log_base = np.log(base)
    cmi = 0.0

    for (xi, yi, zi), c_xyz in xyz_counts.items():
        p_xyz = c_xyz / n
        p_xz = xz_counts[(xi, zi)] / n
        p_yz = yz_counts[(yi, zi)] / n
        p_z = z_counts[zi] / n

        ratio = (p_xyz * p_z) / (p_xz * p_yz)
        cmi += p_xyz * (np.log(ratio) / log_base)

    return max(cmi, 0.0)


def cmi_feature_selection(x, Y, tol=0.05, base=2, verbose=True):
    """
    Greedy feature selection using MI then conditional MI.

    Procedure
    ---------
    1. Start with empty selected set Z
    2. Select the feature j maximizing I(X; Y_j)
    3. Repeatedly select the feature j maximizing I(X; Y_j | Z)
    4. Stop when the maximum score is < tol

    Parameters
    ----------
    x : 1D array, shape (n_samples,)
        Discrete target variable.
    Y : 2D array, shape (n_samples, n_features)
        Discrete candidate features.
    tol : float, default=0.01
        Stopping threshold.
    base : int or float, default=2
        Logarithm base.
    verbose : bool, default=True
        If True, prints progress.

    Returns
    -------
    selected_indices : list of int
        Indices of selected features in order.
    selected_scores : list of float
        MI/CMI score of each selected feature.
    """
    x = _to_1d_object_array(x)
    Y = _to_2d_object_array(Y)

    n_samples, n_features = Y.shape

    if len(x) != n_samples:
        raise ValueError("x and Y must have the same number of samples.")

    remaining = list(range(n_features))
    selected_indices = []
    selected_scores = []

    # First step: ordinary MI
    best_idx = None
    best_score = -np.inf

    if verbose:
        print("Step 1: Mutual information scores")

    for j in remaining:
        score = mutual_information(x, Y[:, j], base=base)
        if verbose:
            print(f"  Feature {j}: MI = {score:.6f}")
        if score > best_score:
            best_score = score
            best_idx = j

    if best_idx is None or best_score < tol:
        if verbose:
            print(f"No feature selected. Best MI = {best_score:.6f} < tol = {tol}")
        return selected_indices, selected_scores

    selected_indices.append(best_idx)
    selected_scores.append(best_score)
    remaining.remove(best_idx)

    if verbose:
        print(f"Selected feature {best_idx} with MI = {best_score:.6f}")

    # Next steps: conditional MI
    step = 2
    while len(remaining) > 0:
        Z = Y[:, selected_indices]

        best_idx = None
        best_score = -np.inf

        if verbose:
            print(f"\nStep {step}: Conditional mutual information scores")

        for j in remaining:
            score = conditional_mutual_information(x, Y[:, j], Z, base=base)
            if verbose:
                print(f"  Feature {j}: CMI = {score:.6f}")
            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx is None or best_score < tol:
            if verbose:
                print(f"Stopping: best CMI = {best_score:.6f} < tol = {tol}")
            break

        selected_indices.append(best_idx)
        selected_scores.append(best_score)
        remaining.remove(best_idx)

        if verbose:
            print(f"Selected feature {best_idx} with CMI = {best_score:.6f}")

        step += 1

    return selected_indices, selected_scores