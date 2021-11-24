#!/usr/bin/env python
import numpy as np
import sys, pickle

from utils.laplace_approximation import constrained_cov
from utils.myGraph import Graph

def generate_data(n, m, g, seed=None, threshold=.5, df=3):
    if seed is not None:
        np.random.seed(seed)

    T = np.random.random((df + n, n))
    C = T.transpose() @ T
    C_star = constrained_cov(g.GetDOL(), C, np.eye(n)) # constrain zeroes of the matrices
    K = np.linalg.inv(C_star)
    triu = np.triu_indices(n, 1)

    if threshold:
        count = 0
        while not ((np.abs(K[triu]) > threshold).astype(int) == g.GetBinaryL()).all():
            T = np.random.random((df + n, n))
            C = T.transpose() @ T
            C_star = constrained_cov(g.GetDOL(), C, np.eye(n)) # constrain zeroes of the matrices
            K = np.linalg.inv(C_star)
            triu = np.triu_indices(n, 1)

            if count > 20:
                raise ValueError("Can't find precision matrix with large enough non-zero values, \
                                try tweaking the threshold parameter.")

    assert(((np.abs(np.linalg.inv(C_star)) > 1e-10)[triu] == g.GetBinaryL()).all()) # zeros at the right places
    assert(np.allclose(C_star, C_star.transpose())) # symmetric
    assert(np.linalg.det(C_star) > 0.) # positive definite

    data = np.random.multivariate_normal(np.zeros(n), C_star, m)
    if m == 1:
        data = data.reshape(1, n)
    return data, K

def get_circle(n):
    g = Graph(n)
    for i in range(1, n):
        g.AddEdge(i - 1, i)
    g.AddEdge(0, n - 1)
    return g

if __name__ == '__main__':
    n, n_obs = 30, 20 #int(sys.argv[1]), int(sys.argv[2])
    gname = 'circle' #sys.argv[3]

    g = get_circle(n)

    data, K = generate_data(n, n_obs, g, threshold=None, seed=0)

    np.savetxt(f"data/circle_precision_{n}.csv", K, delimiter=',')
    np.savetxt(f"data/circle_data_{n}_{n_obs}.csv", data, delimiter=",")
    with open(f'data/graph_circle_{n}.pkl', 'wb') as handle:
        pickle.dump(g, handle)

