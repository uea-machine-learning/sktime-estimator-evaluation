import numpy as np
from aeon.distances import distance
from aeon.transformations.collection.base import BaseCollectionTransformer
from sklearn.utils.random import check_random_state

"""
NOTE to future:

So I have it basically all developed upto line 160:
    X = exactCDmex(nA, nR, nO, X0, lenA, d, fro_norm, 20)

For this function im had to try convert the matlab c code but it's super weird.

I have equality on the first iteration (residue) but subsequent iterations are not equal.
No idea how to fix.

"""
# TODO: Remember to uncomment the stuff in construct_sparse as currently loading from
#  .npy files

# TODO Matrix factorization part which I think is essentially the predict function
# (_transform)


class SPIRAL(BaseCollectionTransformer):
    """SPIRAL clustering estimator."""

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.D = None
        self.Omega = None
        self.d = None
        super().__init__()

    def _fit(self, X, y=None):
        self.random_state = check_random_state(self.random_state)
        n = len(X)
        m = n * 20 * int(np.ceil(np.log(n)))
        if 2 * m > n * n:
            m = n * n // 2

        D, Omega, d = self.construct_sparse(X, n, m)
        self.D = D
        self.Omega = Omega
        self.d = d

    def _transform(self, X, y=None):
        X_0 = np.zeros((len(X), 30), dtype=float)
        return matrix_completion_sparse(self.D, self.d, self.Omega, X_0)

    def construct_sparse(self, X, n, m):
        """
        Generates the kernel matrix.

        Parameters
        ----------
            X (list of np.array): Input time series data.
            n (int): Number of users.
            m (int): Number of pairs to generate.

        Returns
        -------
            D (dict): Kernel matrix.
            Omega (dict): Indices matrix.
            d (np.array): Index array.
        """
        print("Step 1: sample and calculate dtw distance...")

        D = {}
        Omega = {}
        d = np.zeros(n, dtype=int)
        length = X[0].shape[1]
        wsize = int(np.ceil(length / 30))
        wsize = min(max(wsize, 1), 40)

        id2d = self.random_state.choice(n * n, 2 * m, replace=False)
        # np.save('/home/chris/projects/clustering-algo-clones/SPIRAL/id2d.npy', id2d)
        idi = np.floor((id2d - 1) / n).astype(int) + 1
        idj = id2d - n * (idi - 1)

        # This maps to id
        id_mask = idi < idj

        idi = idi[id_mask]
        idi = idi[: (m - n) // 2]
        idj = idj[id_mask]
        idj = idj[: (m - n) // 2]

        idi = idi - 1
        idj = idj - 1

        v = np.zeros((m - n) // 2)
        nrm = np.zeros(n)

        for i in range(n):
            # They have this using dtw in original but the best path will always be
            # the diagonal so we can just use euclidean
            nrm[i] = distance(X[i], np.zeros((1, length)), metric="euclidean")

        # np.save('/home/chris/projects/clustering-algo-clones/SPIRAL/nrm.npy', nrm)
        # Should probably move this out and numba this loop
        # for k in range((m - n) // 2):
        #     i = idi[k]
        #     j = idj[k]
        #     v[k] = (nrm[i] ** 2 + nrm[j] ** 2 - distance(X[i - 1], X[j - 1], metric='dtw', window=0.2)) / (
        #                        2 * nrm[i] * nrm[j])
        # np.save('/home/chris/projects/clustering-algo-clones/SPIRAL/v.npy', v)
        # Load the precomputed values
        v = np.load("/home/chris/projects/clustering-algo-clones/SPIRAL/v.npy")

        col = np.concatenate((idi, idj, np.arange(0, n)))
        row = np.concatenate((idj, idi, np.arange(0, n)))
        v = np.concatenate((v, v, np.ones(n)))

        idx_sort = np.argsort(col)
        col = col[idx_sort]
        row = row[idx_sort]
        v = v[idx_sort]

        col_size = len(col)

        start = 0
        nd = 0
        for i in range(n):
            while True:
                if nd >= col_size or col[nd] != i:
                    break
                nd += 1
            Omega[i] = row[start:nd]
            D[i] = v[start:nd]
            d[i] = np.where(Omega[i] == i)[0][0]
            start = nd

        return D, Omega, d


def matrix_completion_sparse(A, d, Omega, X0):
    print("Step 2: matrix factorization...")
    n = len(A)
    lenA = np.zeros(n, dtype=int)

    for i in range(n):
        lenA[i] = len(A[i])

    m = max(lenA)

    nA = np.zeros((n, m))
    nO = np.zeros((n, m), dtype=int)

    for i in range(n):
        nA[i, : len(A[i])] = A[i]
        nO[i, 0 : len(Omega[i])] = Omega[i] - 1

    d = d - 1
    nR = nA
    k = X0.shape[1]

    fro_norm = np.linalg.norm(nA, "fro")

    np.save("/home/chris/projects/clustering-algo-clones/SPIRAL/nA.npy", nA)
    np.save("/home/chris/projects/clustering-algo-clones/SPIRAL/nR.npy", nR)
    np.save("/home/chris/projects/clustering-algo-clones/SPIRAL/nO.npy", nO)
    np.save("/home/chris/projects/clustering-algo-clones/SPIRAL/X0.npy", X0)
    np.save("/home/chris/projects/clustering-algo-clones/SPIRAL/lenA.npy", lenA)
    np.save("/home/chris/projects/clustering-algo-clones/SPIRAL/d.npy", d)

    X = exactCDmex(nA, nR, nO, X0, lenA, d, fro_norm, 20)
    return X.reshape(X0.shape)


from numba import njit


@njit(cache=True)
def cubic_root(d):
    if d < 0:
        return -np.cbrt(-d)
    else:
        return np.cbrt(d)


@njit(cache=True)
def root_c(a, b, verbose=False):
    if verbose:
        print("+++++++++++++++++++++")
        print(f"Input to root_c: a={a}, b={b}")
    a3 = 4 * (a**3)
    b2 = 27 * (b**2)
    delta = a3 + b2
    if delta <= 0:
        if verbose:
            print("Inside delta <= 0")
        r3 = 2 * np.sqrt(-a / 3)
        th3 = np.arctan2(np.sqrt(-delta / 108), -b / 2) / 3
        ymax = 0
        xopt = 0
        if verbose:
            print(f"r3 = {r3}")
            print(f"th3 = {th3}")
        for k in range(0, 5, 2):
            # x = r3 * np.cos(th3 + (k * np.pi / 3))
            x = r3 * np.cos(th3 + (k * 3.14159265 / 3))
            y = (x**4) / 4 + a * (x**2) / 2 + b * x
            if y < ymax:
                ymax = y
                xopt = x
        if verbose:
            print(f"xopt = {xopt}")
        return xopt
    else:
        if verbose:
            print("Inside delta > 0")
        z = np.sqrt(delta / 27)
        x = cubic_root(0.5 * (-b + z)) + cubic_root(0.5 * (-b - z))
        if verbose:
            print(f"x = {x}")
        return x


@njit
def residue(nR, normA, n, lenA):
    print("+++++++++++++++++++++")
    print(f"normA = {normA}")
    print(f"n = {n}")
    print(f"lenA = {lenA[0:10]}")
    print(f"nR = {nR[0:10]}")
    r = 0
    for i in range(n):
        for j in range(lenA[i]):
            r += nR[j * n + i] * nR[j * n + i]
    print(f"r = {r}")
    print(f"Result = {r / normA / normA}")
    return r / normA / normA


# This doesnt need to be sepeartte function refactro
@njit
def iterations(nA, nR, nO, X, n, m, k, lenA, max_iters, normA, d):
    res = []
    for iter in range(max_iters):
        for i in range(k):
            in_var = i * n
            for t in range(n):
                x = X[in_var + t]
                for j in range(lenA[t]):
                    indexR = j * n + t
                    indexO = in_var + nO[indexR]
                    nR[indexR] += x * X[indexO]
            for j in range(n):
                p = 0
                q = 0
                for t in range(lenA[j]):
                    tn = t * n
                    index = in_var + nO[tn + j]
                    p += X[index] * X[index]
                    q -= X[index] * nR[tn + j]
                p -= X[in_var + j] * X[in_var + j] + nR[j + n * d[j]]
                q += nR[d[j] * n + j] * X[in_var + j]
                X[in_var + j] = root_c(p, q)

            # X isnt being updated
            for t in range(n):
                x = X[in_var + t]
                for j in range(lenA[t]):
                    indexR = j * n + t
                    indexO = in_var + nO[indexR]
                    nR[indexR] -= x * X[indexO]

        res.append(residue(nR, normA, n, lenA))
    return X, res


def exactCDmex(nA, nR, nO, X0, lenA, d, normA, max_iters):
    nA_shape = nA.shape
    X0_shape = X0.shape
    nR = nR.flatten(order="F")
    nO = nO.flatten(order="F")
    X0 = X0.flatten(order="F")
    d = d.flatten(order="F")
    nA = nA.flatten(order="F")
    X, res = iterations(
        nA, nR, nO, X0, nA_shape[0], nA_shape[1], X0_shape[1], lenA, max_iters, normA, d
    )

    # for val in printMe:
    #     print(f"# {val[0]}: i={val[1]}, t={val[2]}, j={val[3]}, nR[j*n+t]={val[4]}")

    for i in range(max_iters):
        print(f"# {i}: residue={res[i]}")

    temp_res = X.reshape(X0_shape)
    return X


if __name__ == "__main__":
    import os

    from aeon.datasets import load_from_tsfile as load_data

    DATASETS_PATH = "/home/chris/Documents/Univariate_ts"
    DATASET = "FiftyWords"
    dataset_path = os.path.join(DATASETS_PATH, DATASET)

    X_train, y_train = load_data(os.path.join(dataset_path, f"{DATASET}_TRAIN.ts"))
    X_test, y_test = load_data(os.path.join(dataset_path, f"{DATASET}_TEST.ts"))
    X = [X_train[i] for i in range(X_train.shape[0])]
    X += [X_test[i] for i in range(X_test.shape[0])]
    X = np.array(X)
    X_labels = np.concatenate([y_train, y_test])

    spiral = SPIRAL(random_state=0)
    train_T = spiral.fit_transform(X)
    # test_T = spiral.transform(X_test)

    from sklearn.cluster import KMeans
    from sklearn.metrics import rand_score

    n_clusters = np.unique(y_train).shape[0]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    train_labels = kmeans.fit_predict(train_T)
    # test_labels = kmeans.predict(test_T)

    print(f"Train Rand Score: {rand_score(X_labels, train_labels)}")
    # print(f"Test Rand Score: {rand_score(y_test, test_labels)}")

    stop = ""
