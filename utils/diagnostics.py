import numpy as np

def ACF(a):
    """
    A function that computes Autocorrelation Function (ACF) from an array.
    """
    n = len(a)
    a_bar = np.mean(a)
    a = a - a_bar # de-mean a
    normalising = np.sum(a * a)

    acf = [ np.sum(a[i:] * a[:n - i]) for i in range(n)]
    return np.array(acf) / normalising

def IAT(a):
    """
    A function that computes Integrated Autocorrelation Time (IAT) from an array.
    Implementation according to https://github.com/LaplacesDemonR/LaplacesDemon/blob/master/R/IAT.R
    """

    if a is None: raise ValueError("None array")
    a = np.array(a)
    n = len(a)
    mu = np.mean(a)
    s2 = np.var(a, ddof=1)

    # Maximum lag is half the sample size
    maxlag = max(3, n // 2)
    # Gammas are sums of two consecutive autocovariances
    Ga = [0,0]
    Ga[0] = s2
    lg = 1
    Ga[0] = Ga[0] + np.sum((a[:n-lg] - mu) *
                            (a[lg:] - mu)) / n
    m = 1
    lg = 2 * m
    Ga[1] = np.sum((a[:n-lg] - mu) * (a[lg:] - mu)) / n
    lg = 2 * m + 1
    Ga[1] = Ga[1] + np.sum((a[:n-lg] - mu) *
                           (a[lg:] - mu)) / n
    IAT = Ga[0] / s2

    while ((Ga[1] > 0) and (Ga[1] < Ga[0])):
        m += 1
        if (2 * m + 1 > maxlag):
            print('Not enough data, maxlag = ' + str(maxlag) + '\n')
            break
        Ga[0] = Ga[1]
        lg = 2 * m
        Ga[1] = np.sum((a[:n-lg] - mu) *
                       (a[lg:] - mu)) / n
        lg = 2 * m + 1
        Ga[1] = Ga[1] + np.sum((a[:n-lg] - mu) *
                               (a[lg:] - mu)) / n
        IAT += Ga[0]/s2

    IAT = -1 + 2*IAT
    return IAT

def str_list_to_adjm(n, str_list):
    """
    A function that takes a list of graph samples in string format and computes its edge inclusion probabilities.
    """
    AdjM = np.zeros((n,n))
    triu = np.triu_indices(n, 1)
    l = np.zeros(triu[0].shape, dtype=int)
    for s in str_list:
        l += np.array(list(s), dtype=int)
    AdjM[triu] = l
    return (AdjM + AdjM.transpose()) / len(str_list)

def str_list_to_median_graph(n, str_list, threshold=.5):
    """
    A function that takes a list of graph samples in string format and computes the median graph estimate.
    """
    return (str_list_to_adjm(n, str_list) > threshold).astype(int)

"""Assumption: for all distances len(s1) == len(s2)"""
def jaccard_distance(s1, s2):
    A = np.array(list(s1), dtype=int)
    B = np.array(list(s2), dtype=int)

    A_size = A.sum()
    B_size = B.sum()
    intersection_size = np.logical_and(A, B).sum()

    if A_size > 0 or B_size > 0:
        return 1 - intersection_size / (A_size + B_size - intersection_size)
    else:
        return 1

def overlap_coeff_distance(s1, s2):
    A = set(np.where([ int(c) > 0 for c in s1])[0])
    B = set(np.where([ int(c) > 0 for c in s2])[0])
    A_cap_B = A.intersection(B)
    if A_cap_B:
        return 1 - len(A_cap_B) / np.min(len(A), len(B))
    else:
        return 1

def hamming_distance(s1, s2):
    return np.sum([ s1[i] != s2[i] for i in range(len(s1)) ])

def size_distance(s1, s2):
    A = np.array(list(s1), dtype=int)
    B = np.array(list(s2), dtype=int)

    return np.abs(A.sum() - B.sum())
