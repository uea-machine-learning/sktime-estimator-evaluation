if __name__ == "__main__":
    from aeon.datasets import load_gunpoint
    from aeon.distances import pairwise_distance
    from sklearn.cluster import AgglomerativeClustering

    X_train_utsc, y_train_utsc = load_gunpoint(return_X_y=True)

    pw_dist = pairwise_distance(X_train_utsc, metric="dtw")

    clst = AgglomerativeClustering(metric="precomputed", linkage="complete")
    clst.fit(pw_dist)
    print(clst.labels_)
