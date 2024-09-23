import numpy as np
from aeon.clustering.averaging import elastic_barycenter_average, mean_average
from aeon.distances import distance as ts_distance
from aeon.distances import pairwise_distance
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y


def calinski_harabasz_score_time_series(
    X,
    labels,
    distance="dtw",
    distance_params=None,
):
    """
    Compute the Calinski and Harabasz score for time series data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_channels, n_timepoints)
        A collection of time series data points.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    distance : str or callable, default='dtw'
        The distance metric to use.
    distance_params : dict, optional
        Additional parameters to pass to the distance function.
    average_function : callable, default=elastic_barycenter_average
        Function to compute the cluster centroids.

    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.
    """
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = X.shape[0]
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    if distance_params is None:
        distance_params = {}

    # Compute the global centroid
    if distance == "euclidean":
        global_centroid = mean_average(X, distance=distance, **distance_params)
    else:
        global_centroid = elastic_barycenter_average(
            X,
            distance=distance,
            **distance_params,
        )

    extra_disp = 0.0
    intra_disp = 0.0

    centroids = []
    cluster_sizes = []

    for k in range(n_labels):
        cluster_k = X[labels == k]
        n_k = cluster_k.shape[0]

        if n_k == 0:
            centroids.append(None)
            cluster_sizes.append(0)
            continue

        # Compute the centroid of cluster k
        if distance == "euclidean":
            centroid_k = mean_average(cluster_k, distance=distance, **distance_params)
        else:
            centroid_k = elastic_barycenter_average(
                cluster_k,
                distance=distance,
                **distance_params,
            )
        centroids.append(centroid_k)
        cluster_sizes.append(n_k)

    # Filter out empty clusters
    valid_indices = [i for i, c in enumerate(centroids) if c is not None]
    valid_centroids = [centroids[i] for i in valid_indices]
    valid_cluster_sizes = [cluster_sizes[i] for i in valid_indices]

    # Compute distances between centroids and global centroid
    # Since centroids may have different shapes, we use a list comprehension
    dist_centroids = np.array(
        [
            ts_distance(c, global_centroid, metric=distance, **distance_params)
            for c in valid_centroids
        ]
    )
    dist_centroids_squared = dist_centroids**2

    # Compute the extra dispersion (between-cluster dispersion)
    extra_disp = np.sum(np.array(valid_cluster_sizes) * dist_centroids_squared)

    # Compute the intra-cluster dispersion
    intra_disp = 0.0
    for idx, k in enumerate(valid_indices):
        cluster_k = X[labels == k]
        centroid_k = valid_centroids[idx]
        n_k = cluster_k.shape[0]

        # Compute distances between cluster_k and centroid_k
        dists = pairwise_distance(
            cluster_k, centroid_k[None, ...], metric=distance, **distance_params
        )
        intra_disp_k = np.sum(dists.squeeze() ** 2)
        intra_disp += intra_disp_k

    score = (
        1.0
        if intra_disp == 0.0
        else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
    )

    return score


def davies_bouldin_score_time_series(
    X,
    labels,
    distance="dtw",
    distance_params=None,
):
    """
    Compute the Davies-Bouldin score for time series data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_channels, n_timepoints)
        A collection of time series data points.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    distance : str or callable, default='dtw'
        The distance metric to use.
    distance_params : dict, optional
        Additional parameters to pass to the distance function.
    average_function : callable, default=elastic_barycenter_average
        Function to compute the cluster centroids.

    Returns
    -------
    score : float
        The resulting Davies-Bouldin score.
    """
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = X.shape[0]
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    if distance_params is None:
        distance_params = {}

    # Compute centroids and intra-cluster dispersions
    centroids = []
    intra_dists = []

    for k in range(n_labels):
        cluster_k = X[labels == k]
        n_k = cluster_k.shape[0]

        if n_k == 0:
            centroids.append(None)
            intra_dists.append(0.0)
            continue

        # Compute centroid
        if distance == "euclidean":
            centroid_k = mean_average(cluster_k, distance=distance, **distance_params)
        else:
            centroid_k = elastic_barycenter_average(
                cluster_k,
                distance=distance,
                **distance_params,
            )

        if len(centroid_k.shape) == 1:
            centroid_k = centroid_k.reshape(1, -1)
        centroids.append(centroid_k)

        # Compute intra-cluster dispersion (mean distance between samples and centroid)
        dists = pairwise_distance(
            cluster_k, centroid_k[None, ...], metric=distance, **distance_params
        )
        intra_disp_k = np.mean(dists.squeeze())
        intra_dists.append(intra_disp_k)

    # Compute distances between centroids
    # Prepare a list of valid centroids (clusters that are not empty)
    valid_indices = [i for i, c in enumerate(centroids) if c is not None]
    valid_centroids = [centroids[i] for i in valid_indices]
    n_valid = len(valid_centroids)

    # Initialize centroid distance matrix
    centroid_distances = np.full((n_labels, n_labels), np.inf)

    # Compute pairwise distances between centroids
    if n_valid > 1:
        # Convert centroids to array if possible
        # Since centroids might have varying lengths/shapes, we use a list comprehension
        for i in valid_indices:
            for j in valid_indices:
                if i < j:
                    try:
                        dist = ts_distance(
                            centroids[i],
                            centroids[j],
                            metric=distance,
                            **distance_params,
                        )
                        centroid_distances[i, j] = dist
                        centroid_distances[j, i] = dist
                    except:
                        stop = ""

    # Compute the Davies-Bouldin score
    scores = np.zeros(n_labels)

    for i in valid_indices:
        ratios = []
        for j in valid_indices:
            if i != j:
                numerator = intra_dists[i] + intra_dists[j]
                denominator = centroid_distances[i, j]
                if denominator != 0 and not np.isinf(denominator):
                    ratios.append(numerator / denominator)
        if ratios:
            scores[i] = max(ratios)
        else:
            scores[i] = 0.0

    return np.mean(scores[scores != 0])


if __name__ == "__main__":
    from aeon.clustering.averaging import elastic_barycenter_average
    from aeon.distances import pairwise_distance
    from aeon.testing.data_generation import make_example_3d_numpy
    from sklearn.metrics.cluster import calinski_harabasz_score, davies_bouldin_score

    # Generate sample 3D time series data
    X, y_true = make_example_3d_numpy(100, 1, 20, random_state=0, return_y=True)
    sklearn_X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

    # Assume 'labels' are the predicted cluster labels for X
    # For demonstration, we'll generate random labels
    import numpy as np

    np.random.seed(0)
    labels = np.random.randint(0, 3, size=100)  # Assuming 3 clusters

    # Distance metric and parameters
    distance_string = "msm"

    # Compute Calinski-Harabasz score
    ch_score = calinski_harabasz_score_time_series(
        X,
        labels,
        distance=distance_string,
    )

    sklearn_ch_score = calinski_harabasz_score(sklearn_X, labels)

    # Compute Davies-Bouldin score
    db_score = davies_bouldin_score_time_series(
        X,
        labels,
        distance=distance_string,
    )
    sklearn_db_score = davies_bouldin_score(sklearn_X, labels)

    print(f"Elastic Calinski-Harabasz Score: {ch_score}")
    print(f"Elastic Davies-Bouldin Score: {db_score}")
    print(f"Sklearn Calinski-Harabasz Score: {sklearn_ch_score}")
    print(f"Sklearn Davies-Bouldin Score: {sklearn_db_score}")
