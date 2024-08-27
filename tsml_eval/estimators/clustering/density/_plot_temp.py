import os

import matplotlib.gridspec as gridspec
import numpy as np
from aeon.clustering import (
    TimeSeriesCLARA,
    TimeSeriesCLARANS,
    TimeSeriesKMeans,
    TimeSeriesKMedoids,
    TimeSeriesKShapes,
)
from aeon.clustering.averaging import (
    elastic_barycenter_average,
    mean_average,
    subgradient_barycenter_average,
)
from aeon.datasets import load_from_tsfile
from matplotlib import pyplot as plt

from tsml_eval.estimators.clustering import ElasticSOM, KSpectralCentroid
from tsml_eval.estimators.clustering.density._meanshift import TimeSeriesMeanShift

# Define the path and dataset name


def ksc_plotting(centres, X_train, sz):
    for i in range(centres.shape[0]):  # Iterate over the number of clusters
        max_val = np.max(centres[i])
        centres[i] = 1.2 * centres[i] / max_val  # Scale to 1.2 times the max value

    fig = plt.figure(figsize=(15, 5))  # Adjust the figure size as needed
    gs = plt.GridSpec(1, 3, wspace=0.3)  # Use GridSpec for better layout control

    cluster_colors = ["#FF0000", "#3749EF", "#99FF80"]  # Colors for the other values

    for yi in range(3):
        ax = fig.add_subplot(gs[0, yi])
        for xx in X_train[y_pred == yi]:
            ax.plot(xx.ravel(), color="black", alpha=0.2)
        ax.plot(centres[yi].ravel(), color=cluster_colors[yi], linewidth=2)
        ax.set_xlim(0, sz)
        ax.set_ylim(-4, 4)
        ax.set_title(f"Cluster {yi + 1}")  # Add title to each graph

    plt.tight_layout()
    # Get name of clustering method
    method = km.__class__.__name__
    plt.savefig(f"{method}-partition-cluster.pdf")
    plt.show()


cluster_colors = [
    "#FF5733",  # Bright Red-Orange
    "#33FF57",  # Bright Green
    "#3357FF",  # Bright Blue
    "#FF33A6",  # Bright Pink
    "#33FFF6",  # Bright Cyan
    "#FFA633",  # Bright Orange
    "#A633FF",  # Bright Purple
    "#33FFDD",  # Bright Aqua
    "#FF3357",  # Bright Coral
    "#57FF33",  # Bright Lime Green
    "#5733FF",  # Bright Indigo
    "#FF33F6",  # Bright Magenta
    "#33FFA6",  # Bright Mint
    "#FF33C4",  # Bright Rose
    "#C433FF",  # Bright Violet
    "#33FF85",  # Bright Turquoise
    "#FF5733",  # Bright Tomato
    "#5733FF",  # Bright Ultra Violet
    "#33FF99",  # Bright Spring Green
    "#FF33B5",  # Bright Raspberry
]


def normal_plotting(centres, X_train, sz):
    num_centres = len(centres)
    num_cols = 3
    num_rows = (
        num_centres + num_cols - 1
    ) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(15, 5 * num_rows)
    )  # Adjust the figure size as needed
    axes = (
        axes.flatten()
    )  # Flatten the 2D array of axes to a 1D array for easier iteration

    for yi in range(num_centres):
        ax = axes[yi]
        for xx in X_train[y_pred == yi]:
            ax.plot(xx.ravel(), color="black", alpha=0.2)
        # ax.plot(centres[yi].ravel(), color=cluster_colors[yi % len(cluster_colors)],
        #         linewidth=2)
        ax.set_xlim(0, sz)
        ax.set_ylim(-4, 4)
        ax.set_title(f"Cluster {yi + 1}")  # Add title to each graph

    # Hide any unused subplots
    for j in range(num_centres, num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # Get name of clustering method
    # method = km.__class__.__name__
    # plt.savefig(f'{method}-partition-cluster.pdf')
    plt.show()


dataset_name = "ACSF1"

if __name__ == "__main__":
    DATASET_PATH = "/Users/chris/Documents/Phd-data/Datasets/Univariate_ts"
    X_train, y_train = load_from_tsfile(
        f"{DATASET_PATH}/{dataset_name}/{dataset_name}_TRAIN.ts"
    )
    sz = X_train.shape[-1]
    n_clusters = len(set(y_train))
    # temp = TimeSeriesKMeans(distance="euclidean", averaging_method="ba", n_clusters=3)
    # temp.fit(X_train)
    km = TimeSeriesMeanShift(distance="euclidean", averaging_method="mean")
    y_pred = km.fit_predict(X_train)

    centres = km.cluster_centers_
    print(len(centres))

    if isinstance(km, TimeSeriesKShapes):
        centres = km._tslearn_k_shapes.cluster_centers_
        centres = centres.swapaxes(1, 2)
    else:
        centres = km.cluster_centers_

    if isinstance(km, KSpectralCentroid):
        ksc_plotting(centres, X_train, sz)
    else:
        normal_plotting(centres, X_train, sz)

    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
        rand_score,
    )

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"\nAMI {adjusted_mutual_info_score(y_pred, y_train)}")
    print(f"NMI {normalized_mutual_info_score(y_pred, y_train)}")
    print(f"ARI {adjusted_rand_score(y_pred, y_train)}")
    print(f"Rand {rand_score(y_pred, y_train)}")
