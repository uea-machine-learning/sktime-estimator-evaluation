from collections import Counter

import numpy as np


def unbalance_data(X, y, target_class, n_samples):
    """Keep only n_samples of the target_class, retain all other classes."""
    print("Original dataset shape %s" % Counter(y))  # noqa: T201
    if n_samples is None:
        return X, y
    mask = y == target_class
    target_indices = np.where(mask)[0]
    other_indices = np.where(~mask)[0]

    # Ensure there are enough samples in the target class
    if len(target_indices) == 0:
        raise ValueError(f"Target class {target_class} not found in y.")
    if len(target_indices) < n_samples:
        raise ValueError(
            f"Not enough samples in class {target_class}. "
            f"Requested {n_samples}, but only {len(target_indices)} available."
        )

    # Select the desired number of samples from the target class
    reduced_target_indices = np.random.choice(target_indices, n_samples, replace=False)

    # Combine the reduced target class samples with all other class samples
    new_indices = np.concatenate([reduced_target_indices, other_indices])
    print("Unbalanced dataset shape %s" % Counter(y[new_indices]))  # noqa: T201
    return X[new_indices], y[new_indices]


def _make_samples(
    X, random_state, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, y=None
):
    """Create artificial samples constructed along the line connecting neighbours.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_dims, n_timepoints)
        Points from which the synthetic points will be created.
    random_state : RandomState
        A random number generator instance to make results reproducible.
    y_dtype : dtype
        The data type of the targets.
    y_type : str or int
        The minority target value, just so the function can return the
        target values for the synthetic variables with correct length in
        a clear format.
    nn_data : ndarray of shape (n_samples_all, n_dims, n_timepoints)
        Dataset containing all the neighbours to be used.
    nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
        Indices of the nearest neighbours for each sample in `nn_data`.
    n_samples : int
        The number of samples to generate.
    step_size : float, default=1.0
        The step size to create synthetic samples.
    y : ndarray of shape (n_samples_all,), default=None
        The true target associated with `nn_data`. Used by Borderline SMOTE-2 to
        weight the distances in the sample generation process.

    Returns
    -------
    X_new : ndarray of shape (n_samples, n_dims, n_timepoints)
        Synthetically generated samples.
    y_new : ndarray of shape (n_samples,)
        Target values for synthetic samples.
    """
    samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

    # Generate random steps for interpolation
    steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis, np.newaxis]

    # Determine rows and columns for the nearest neighbours
    rows = np.floor_divide(samples_indices, nn_num.shape[1])
    cols = np.mod(samples_indices, nn_num.shape[1])

    # Generate synthetic samples by interpolating between points
    X_base = X[rows]
    X_neighbor = nn_data[nn_num[rows, cols]]
    X_new = X_base + steps * (X_neighbor - X_base)

    y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
    return X_new, y_new
