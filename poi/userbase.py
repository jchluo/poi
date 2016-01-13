# -*- coding: utf-8 -*-

import time
import logging
import math

import numpy as np
from scipy.sparse import csr_matrix

from .models import Recommender 
from .utils import nonzero 

__all__ = ["UserBase"]

log = logging.getLogger(__name__)

def similarity(arri, arrj):
    """Cos similarity.
    usage:
     >>> from scipy.sparse import csr_matrix 
     >>> a = csr_matrix([[2,1,1]])
     >>> b = csr_matrix([[1,0,1]])
     >>> print round(similarity(a, b), 5)
     0.28868
    """
    s = arri.multiply(arrj).sum()
    if s == 0:
        return 0.0
    m = np.sum(arri.data ** 2) * np.sum(arrj.data ** 2)
    alpha = min(len(nonzero(arri, 0) & nonzero(arrj, 0)) / 6.0, 1.0)
    return float(s) / math.sqrt(m) * alpha 


class UserBase(Recommender):
    def __init__(self, checkins, num_neighbor=10):
        super(UserBase, self).__init__(checkins);
        self.num_neighbor = num_neighbor
        self._neighbors = None 
        self.between = np.zeros((self.num_users, self.num_users)) 

    def similarity(self):
        t0 = time.time()
        for ui in xrange(self.num_users):
            for uj in xrange(ui + 1, self.num_users):
                s = similarity(self.matrix[ui], self.matrix[uj])
                self.between[uj, ui] = s
                self.between[ui, uj] = s
            if ui % 200 == 0:
                t1 = time.time()
                log.debug("similarity user: %i(%.f%%) time: %.2f" % \
                        (ui, ui * 100.0 / self.num_users, t1 - t0))
        self.between = csr_matrix(self.between)

    def neighbors(self, num_neighbor=None):
        t1  = time.time()
        if num_neighbor is not None:
            self.num_neighbor = num_neighbor

        self._neighbors = {}
        for u in xrange(self.num_users):
            beg = self.between.indptr[u] 
            end = self.between.indptr[u + 1] 
            cols = self.between.indices[beg: end]
            friends = [(c, self.between[u, c]) for c in cols]
            friends.sort(key=lambda x: x[1], reverse=True)
            self._neighbors[u] = [f for f, s in friends[: self.num_neighbor]]
        t2  = time.time()
        log.debug("neighbors N: %i time: %.2f" % (self.num_neighbor, t2 - t1))
        return self._neighbors 

    def predict(self, user, item):
        if self._neighbors is None:
            raise Exception("please call neighbors() method before predict().")
        if self.matrix[user, item] > 0:
            return self.matrix[user, item]

        s = 0.0
        w = 0.0
        for u in self._neighbors[user]:
            w += self.between[user][u]
            if self.matrix[u, item] > 0:
                s += self.between[user][u]
        if w > 0.0:
            return s / w
        return 0.0

