# -*- coding: utf-8 -*-
import time
import logging
import math

import numpy as np
from .models import Recommender 

__all__ = ["UserBase"]

log = logging.getLogger(__name__)

def similarity(di, dj):
    """Cos similarity.
    usage:
     >>> a = {0:2, 1:1, 2:1}
     >>> b = {0:1, 2:1}
     >>> print round(similarity(a, b), 5)
     0.86603
    """
    inds = set(di.keys()) & set(dj.keys()) 
    s = len(set(di.keys()) & set(dj.keys())) 
    if s == 0:
        return 0.0

    up = sum([di[k] * dj[k] for k in inds]) 
    m = sum([di[k] ** 2 for k in di]) * sum([dj[k] ** 2 for k in dj]) 
    #alpha = min(s / 6.0, 1.0)
    alpha = 1.0
    return float(up) / math.sqrt(m) * alpha 


class SSMatrix(object):
    """Symmetry Sparse Matrix.
    usage:
     >>> m = SSMatrix(3, 3)
     >>> m[0, 0] = 1 
     >>> m[0, 1] = 2
     >>> print m[1, 0], m[0, 0], m[1, 1]
     2 1 0.0
     >>> print m.items(1)
     [(0, 2), (1, 0.0), (2, 0.0)]
    """
    def __init__(self, num_row, num_col):
        self._data = {}
        self.num_row = num_row
        self.num_col = num_col

    def __getitem__(self, key):
        row, col = key
        if row in self._data:
            if col in self._data[row]:
                return self._data[row][col]

        if col in self._data:
            if row in self._data[col]:
                return self._data[col][row]
        return 0.0

    def __setitem__(self, key, val):
        if val == 0.0:
            return None
        row, col = key
        if row >= self.num_row:
            raise ValueError("out of row boundary.")
        if col >= self.num_col:
            raise ValueError("out of col boundary.")
        if row not in self._data:
            self._data[row] = {col : val}
        else:
            self._data[row][col] = val

    def items(self, row):
        l = []
        for i in xrange(self.num_col):
            l.append((i, self.__getitem__((row, i))))
        return l


class UserBase(Recommender):
    """User base K nearest neighbors algorithm.
       usage:
        >>> cks = {0: [1,2,3], 1: [0, 1, 2], 2:[1, 2]}
        >>> ub = UserBase(cks)
        >>> ub.similarity()
        >>> ub.neighbors(1)    # only concern 1 neighbors
        >>> ub.recommend(0, 1) #recommend 1 item for user 0
        [0]
    """
    def __init__(self, checkins, num_neighbor=10):
        super(UserBase, self).__init__(checkins);
        self.num_neighbor = num_neighbor
        self._neighbors = None 
        self.between = SSMatrix(self.num_items, self.num_items) 

    def __repr__(self):
        return "<UserBase [K=%i]>" % self.num_neighbor

    def similarity(self):
        t0 = time.time()
        for ui in xrange(self.num_users):
            for uj in xrange(ui + 1, self.num_users):
                s = similarity(self.checkins[ui], self.checkins[uj])
                self.between[ui, uj] = s

            if ui % 200 == 0:
                t1 = time.time()
                log.debug("similarity user: %i(%.f%%) time: %.2fs" % \
                        (ui, ui * 100.0 / self.num_users, t1 - t0))

    def neighbors(self, num_neighbor=None):
        t1  = time.time()
        if num_neighbor is not None:
            self.num_neighbor = num_neighbor

        self._neighbors = {}
        for u in xrange(self.num_users):
            friends = self.between.items(u) 
            friends.sort(key=lambda x: x[1], reverse=True)
            self._neighbors[u] = [f for f, s in friends[: self.num_neighbor]]
        t2  = time.time()
        log.debug("neighbors N: %i time: %.2fs" % (self.num_neighbor, t2 - t1))

    def predict(self, user, item):
        if self._neighbors is None:
            raise Exception("please call neighbors() method before predict().")
        if user in self.checkins:
            if item in self.checkins[user]:
                return self.checkins[user][item]

        s = 0.0
        w = 0.0
        for u in self._neighbors[user]:
            w += self.between[user, u]
            if item in self.checkins[u]:
                s += self.between[user, u]
        if w > 0.0:
            return s / w
        return 0.0

