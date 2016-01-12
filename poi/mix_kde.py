
import numpy as np

def _em(X, eps=0.001):
    """ EM algorithm, find weight.
    X : numpy two dim ndarray.
    return: weights
    usage:
     >>> X = np.array([[1, 2], [2, 4], [3, 1]])
     >>> print em(X)
     [ 0.33586597  0.66413403]
    """
    N, K = X.shape
    # init 
    W = X.sum(axis=0) / float(X.sum())
    # solve
    while True:
        W0 = W
        # E step
        Y = np.tile(W, (N, 1)) * X 
        Q = Y / np.tile(Y.sum(axis=1), (K, 1)).T
        # M step
        W = Q.sum(axis=0) / N
        # termination ?
        if np.fabs(W - W0).sum() < eps:
            break
    return W
     

class MixKDE(object):
    pass
