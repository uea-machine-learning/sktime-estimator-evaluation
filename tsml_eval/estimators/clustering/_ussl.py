import numpy as np
from aeon.clustering.base import BaseClusterer
from scipy.spatial.distance import euclidean
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import StandardScaler

class USSL(BaseClusterer):

    _tags = {
        "capability:multivariate": False,
    }

    def __init__(
            self,
            n_clusters: int = 2,
            min_shapelet_length: int = 5,
            n_equal_length_shapelets: int = 2,
            n_scales_of_shapelet_length: int = 2,
            alpha: int = -100,
            sigma: int = 1,
            n_iters: int = 20,
            eta: float = 0.01,
            epsilon: float = 0.1,
            lambda_1: int = 100,
            lambda_2: int = 10000,
            lambda_3: int = 10000,
            lambda_4: int = 10000,
            random_state: int = None
    ):
        self.min_shapelet_length = min_shapelet_length
        self.n_equal_length_shapelets = n_equal_length_shapelets
        self.n_scales_of_shapelet_length = n_scales_of_shapelet_length
        self.alpha = alpha
        self.sigma = sigma
        self.n_iters = n_iters
        self.eta = eta
        self.epsilon = epsilon
        self.random_state = random_state
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4

        self._random_state = None

        self.labels_ = None
        self.cluster_centers_ = None
        self.learned_shapelets_ = None
        super().__init__(n_clusters=n_clusters)

    def _fit(self, X, y=None):
        self._random_state = check_random_state(self.random_state)
        segment_matrix = self._initiliasation_s(X)
        distance_matrix, _ = self._distance_timeseries_shapelet(X, segment_matrix)
        cluster_centres, distance_matrix_to_centres = self._ussl_weird_kmeans(distance_matrix)

        inverse = np.vstack((-cluster_centres[0, :], cluster_centres[1:, :]))

        W_tp1 = inverse
        S_tp1 = segment_matrix
        GY_tp1 = distance_matrix_to_centres
        gap = 100
        F_tp1 = 10000
        F_t = F_tp1+10**10
        wh_time = 0

        while gap > self.epsilon:
            # Calculation matrix
            X_tp1, Xkj_tp1_skl = self._distance_timeseries_shapelet(X, S_tp1)  # update X_tp1
            L_G_tp1, G_tp1 = self.spectral_timeseries_similarity(X_tp1)  # update L_G_tp1
            SS_tp1, XS_tp1, SSij_tp1_sil = self.shapelet_similarity(S_tp1)  # update SS_tp1
            part1 = 0.5 * self.lambda_1 * np.trace(GY_tp1 @ L_G_tp1 @ GY_tp1.T)
            part2 = 0.5 * self.lambda_4 * np.trace(SS_tp1.T @ SS_tp1)
            part3 = 0.5 * self.lambda_2 * np.trace(
                (W_tp1.T @ X_tp1 - GY_tp1).T @ (W_tp1.T @ X_tp1 - GY_tp1))
            part4 = 0.5 * self.lambda_3 * np.trace(W_tp1.T @ W_tp1)
            F_tp1 = part1 + part2 + part3 + part4

            gap = F_t - F_tp1
            if np.isnan(F_tp1):
                break
            W_tp1 = self.update_W(X_tp1, GY_tp1)
            W_tp1 = self.z_regularization(W_tp1)
            GY_tp1 = self.update_GY(W_tp1, X_tp1, GY_tp1, L_G_tp1)
            S_tp1 = self.update_S(GY_tp1, X_tp1, W_tp1, G_tp1, S_tp1, Xkj_tp1_skl, SSij_tp1_sil, SS_tp1)
            S_tp1 = np.hstack([S_tp1[:, [0]], self.z_regularization(S_tp1[:, 1:])])

            F_t = F_tp1
            wh_time += 1

            if wh_time == 15:
                break
        W_star = W_tp1
        S_star = S_tp1
        mY, nY = GY_tp1.shape
        Y_star = np.zeros(nY, dtype=int)
        for j in range(nY):
            y_index = np.argmax(GY_tp1[:, j])
            Y_star[j] = y_index
        self.labels_ = Y_star
        self.cluster_centers_ = W_star
        self.learned_shapelets_ = S_star

    def _predict(self, X, y=None) -> np.ndarray:
        # I made this predict up I have no idea if it's intended
        segment_matrix = self._initiliasation_s(X)
        distance_matrix, _ = self._distance_timeseries_shapelet(X, segment_matrix)
        _, distance_matrix_to_centres = self._ussl_kmeans_assign(distance_matrix, self.cluster_centers_)
        GY_tp1 = distance_matrix_to_centres

        mY, nY = GY_tp1.shape
        Y_star = np.zeros(nY, dtype=int)
        for j in range(nY):
            y_index = np.argmax(GY_tp1[:, j])
            Y_star[j] = y_index
        return Y_star

    def _score(self, X, y=None):
        raise NotImplementedError("This method is not implemented")

    def derivation_of_S(self, Y, X, W, ST_t, Shape_t, Xkj_skl, SSij_sil, SS):
        DShape_t = Shape_t[:, 1:]
        mST, nST = ST_t.shape
        mDShape, nDShape = DShape_t.shape

        Part1 = np.zeros((mDShape, nDShape))
        Part2 = np.zeros((mDShape, nDShape))
        Part3 = np.zeros((mDShape, nDShape))
        STij_skl = np.zeros((mST, nST, mDShape, nDShape))

        parameter1 = self.lambda_1 / 2 * Y.T @ Y
        parameter2 = self.lambda_4 * SS

        for k in range(mDShape):
            len_k = int(Shape_t[k, 0])
            for l in range(len_k):
                for i in range(mST):
                    for j in range(nST):
                        STij_skl[i, j, k, l] = ST_t[i, j] * (-2 / self.sigma**2) * (X[k, i] - X[k, j]) * (Xkj_skl[k, i, l] - Xkj_skl[k, j, l])
                    P1 = np.zeros((mST, nST))
                    for j in range(nST):
                        if j == i:
                            P1[i, j] = parameter1[i, j] * np.sum(STij_skl[i, :, k, l])
                        else:
                            P1[i, j] = parameter1[i, j] * STij_skl[i, j, k, l]
                Part1[k, l] = np.sum(P1)
                Part2[k, l] = 2 * np.sum(parameter2[k, :] * SSij_sil[k, :, l]) - parameter2[k, k] * SSij_sil[k, k, l]
        mY, nY = Y.shape
        parameter3 = self.lambda_2 * (W.T @ X - Y)
        for k in range(mDShape):
            len_k = int(Shape_t[k, 0])
            for l in range(len_k):
                pa3 = np.zeros((mY,))
                for i in range(mY):
                    P3 = np.zeros((nY,))
                    for j in range(nY):
                        P3[j] = parameter3[i, j] * W[k, i] * Xkj_skl[k, j, l]
                    pa3[i] = np.sum(P3)
                Part3[k, l] = np.sum(pa3)

        Deri_of_S = Part1 + Part2 + Part3
        return Deri_of_S

    def update_W(self, X_t, Y_t):
        mX, nX = X_t.shape
        P1 = self.lambda_2 * X_t @ X_t.T + self.lambda_3 * np.eye(mX)
        P2 = self.lambda_2 * X_t @ Y_t.T
        W_tp1 = np.linalg.inv(P1) @ P2
        return W_tp1

    def z_regularization(self, X):
        d_max = np.max(X)
        d_min = np.min(X)
        return (X - d_min) / (d_max - d_min)

    def update_GY(self, W_tp1, X_t, G_t, L_Gt):
        P1 = self.lambda_2 * W_tp1.T @ X_t
        mL, nL = L_Gt.shape
        P2 = (self.lambda_1 * L_Gt + self.lambda_2 * np.eye(mL)) @ X_t.T @ W_tp1
        GY_tp1_hat = G_t * np.sqrt(P1 / P2.T)
        return GY_tp1_hat

    def update_S(self, Y, X, W, ST_t, S_t, Xkj_skl, SSij_sil, SS):
        DS_t = S_t[:, 1:]
        for _ in range(self.n_iters):
            deri_of_S = self.derivation_of_S(Y, X, W, ST_t, S_t, Xkj_skl, SSij_sil, SS)
            DS_t -= self.eta * deri_of_S
            S_t = np.hstack([S_t[:, [0]], DS_t])
        return S_t

    def shapelet_similarity(self, S):
        mS, nS = S.shape
        DS = S[:, 1:]
        Hij_sil = np.zeros((mS, mS, nS - 1))
        XS = np.zeros((mS, mS))
        H = np.zeros((mS, mS))

        for i in range(mS):
            length_s = int(S[i, 0])
            sh_s = DS[i, :length_s]  # the i-th shapelet
            for j in range(i, mS):
                length_l = int(S[j, 0])
                sh_l = DS[j, :length_l]  # the j-th shapelet
                XS[i, j], XSij_si = self._distance_longseries_shortseries(sh_l.reshape(1, -1), sh_s)  # calculate distance

                # calculate the similarity matrix of shapelets
                H[i, j] = np.exp(-XS[i, j] ** 2 / self.sigma ** 2)
                XS[j, i] = XS[i, j]
                H[j, i] = H[i, j]

                # calculate the derivative of H_(ij) on S_(il)
                Hij_sil[i, j, :length_s] = H[i, j] * (
                            -2 / self.sigma ** 2 * XS[i, j]) * XSij_si
                Hij_sil[j, i, :length_s] = Hij_sil[i, j, :length_s]

        return H, XS, Hij_sil

    def spectral_timeseries_similarity(self, X):
        n_timepoints = X.shape[1]
        D_G = np.zeros((n_timepoints, n_timepoints))
        G = np.zeros((n_timepoints, n_timepoints))

        for j in range(n_timepoints):
            for h in range(j, n_timepoints):
                g = np.linalg.norm(X[:, j] - X[:, h])
                G[j, h] = np.exp(
                    - g ** 2 / self.sigma ** 2)  # Similarity between the j-th time series and the h-th time series
                G[h, j] = G[j, h]
            D_G[j, j] = np.sum(G[j, :])

        L_G = D_G - G  # Laplacian matrix of G; Spectral analysis
        return L_G, G

    def _initiliasation_s(self, X: np.ndarray):
        S_0 = np.zeros((self.n_equal_length_shapelets * self.n_scales_of_shapelet_length,
                        1 + self.n_scales_of_shapelet_length * self.min_shapelet_length))

        for j in range(1, self.n_scales_of_shapelet_length + 1):
            length = j * self.min_shapelet_length
            segment_matrix = self._segment_obtain(X, length)
            S = np.ones((self.n_equal_length_shapelets, 1)) * np.mean(segment_matrix, axis=0)
            start_idx = (j - 1) * self.n_equal_length_shapelets
            end_idx = j * self.n_equal_length_shapelets
            S_0[start_idx:end_idx, 0] = length * np.ones(self.n_equal_length_shapelets)
            S_0[start_idx:end_idx, 1:length+1] = S
        return S_0

    def _segment_obtain(self, X, length):
        n_cases, n_channels, n_timepoints = X.shape
        max_timepoint = n_timepoints - length + 1
        segment_matrix = np.zeros((n_cases * max_timepoint, n_channels, length))
        for instance in range(n_cases):
            for start_timepoint in range(max_timepoint):
                end_timepoint = start_timepoint + length
                seg = X[instance, :, start_timepoint:end_timepoint]
                segment_matrix[instance * max_timepoint + start_timepoint] = seg
        return segment_matrix
    
    def _distance_timeseries_shapelet(self, X, segment_matrix):
        n_cases, n_channels, n_timepoints = X.shape
        segment_matrix_data = segment_matrix[:, 1:]
        segment_cases, segment_timepoints = segment_matrix_data.shape
        dist_derivative_matrix = np.zeros((segment_cases, n_cases, segment_timepoints))
        distance_matrix = np.zeros((segment_cases, n_cases))
        for j in range(n_cases):
            for k in range(segment_cases):
                len_k = int(segment_matrix[k, 0])
                shapelet = segment_matrix_data[k, 0:len_k]
                distance, dist_derivative = self._distance_longseries_shortseries(X[j], shapelet)
                distance_matrix[k, j] = distance
                dist_derivative_matrix[k, j, :len_k] = dist_derivative

        return distance_matrix, dist_derivative_matrix

    def _distance_longseries_shortseries(self, ts, candidate_shapelet):
        candidate_length = len(candidate_shapelet)
        num_seg = ts.shape[1] - candidate_length + 1
        dist_1 = np.zeros((num_seg, candidate_length))
        dist_2 = np.zeros(num_seg)

        for i in range(num_seg):
            seg = ts[:, i:i + candidate_length]
            dist_2[i] = np.sum((candidate_shapelet - seg) ** 2) / candidate_length
            dist_1[i] = (candidate_shapelet - seg) / candidate_length

        exp_alpha_D2 = np.exp(self.alpha * dist_2)
        dist_upper = np.sum(dist_2 * exp_alpha_D2)
        dist_lower = np.sum(exp_alpha_D2)
        dist = dist_upper / dist_lower  # the distance between the series_long and series_short

        dist_derivative = np.zeros(candidate_length)
        for candidate_idx in range(candidate_length):
            inverse_dist_lower_squared = 1 / dist_lower ** 2
            exp_alpha_D2_term = exp_alpha_D2 * (dist_lower * (
                        1 + self.alpha * dist_2) - self.alpha * dist_upper)
            dist_derivative[candidate_idx] = inverse_dist_lower_squared * np.sum(
                dist_1[:, candidate_idx] * exp_alpha_D2_term)

        return dist, dist_derivative

    def _ussl_weird_kmeans(self, X):
        """Ill be honest I dont understand why/how this kmeans works.

        It does something weird by only using one value from each time series to
        represent it ??. I don't know why this is done, but I will try to understand
        it later.
        """
        n_cases, n_timepoints = X.shape
        epsilon = 0.001

        # Initialize centroids
        Xmax = np.max(X)
        cluster_centers = Xmax * self._random_state.rand(n_cases, self.n_clusters)
        v = 2 * epsilon

        while v > epsilon:
            Y, distance_matrix = self._ussl_kmeans_assign(X, cluster_centers)
            new_centers = np.zeros_like(cluster_centers)
            for i in range(self.n_clusters):
                index2 = np.where(Y[i, :] == 1)[0]
                if len(index2) == 0:
                    index2 = self._random_state.randint(0, n_timepoints)
                new_centers[:, i] = np.mean(X[:, index2], axis=-1)

            v = np.trace((new_centers - cluster_centers).T @ (new_centers - cluster_centers))
            cluster_centers = new_centers

        return new_centers,  distance_matrix

    def _ussl_kmeans_assign(self, X, cluster_centers):
        n_timepoints = X.shape[-1]
        Y = np.zeros((self.n_clusters, n_timepoints))
        distance_matrix = np.zeros((self.n_clusters, n_timepoints))
        for i in range(n_timepoints):
            curr_ts_values = X[:, i]
            for j in range(self.n_clusters):
                curr_cluster = cluster_centers[:, j]
                distance_matrix[j, i] = euclidean(curr_ts_values, curr_cluster)
            index = np.argmin(distance_matrix[:, i])
            Y[index, i] = 1

        return Y, distance_matrix


if __name__ == "__main__":
    from aeon.datasets import load_from_tsfile
    from sklearn.metrics import rand_score
    DATA_PATH = "/home/chris/Documents/Univariate_ts"
    DATASET = "ACSF1"
    COMBINE = True
    train_path = f"{DATA_PATH}/{DATASET}/{DATASET}_TRAIN.ts"
    test_path = f"{DATA_PATH}/{DATASET}/{DATASET}_TEST.ts"

    X_train, y_train = load_from_tsfile(train_path)
    X_test, y_test = load_from_tsfile(test_path)

    # Remove middle dim
    X_train = X_train[:, 0, :]
    X_test = X_test[:, 0, :]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if COMBINE:
        X_train = np.vstack((X_train, X_test))
        y_train = np.hstack((y_train, y_test))
    n_clusters = np.unique(y_train).shape[0]
    print(f"Number of clusters: {n_clusters}")
    clusterer = USSL(random_state=1, n_clusters=n_clusters)
    clusterer.fit(X_train)
    temp = clusterer.labels_
    train_score = rand_score(y_train, clusterer.labels_)
    print(f"Train Labels: {temp}")
    print(f"Train score: {train_score}")

    if COMBINE:
        predictions = clusterer.predict(X_test)
        test_score = rand_score(y_test, predictions)
        print(f"Test Labels: {predictions}")
        print(f"Test score: {test_score}")



