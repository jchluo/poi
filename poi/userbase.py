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
     0.28868
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


class UserBase(Recommender):
    def __init__(self, checkins, num_neighbor=10):
        super(UserBase, self).__init__(checkins);
        self.num_neighbor = num_neighbor
        self._neighbors = None 
        self.between = {} 

    def __repr__(self):
        return "<UserBase [K=%i]>" % self.num_neighbor

    def similarity(self):
        t0 = time.time()
        for ui in xrange(self.num_users):
            if ui not in self.between:
                self.between[ui] = {} 
            self.between[ui][ui] = 0.0 
            for uj in xrange(ui + 1, self.num_users):
                if uj not in self.between:
                    self.between[uj] = {} 

                s = similarity(self.checkins[ui], self.checkins[uj])
                self.between[uj][ui] = s
                self.between[ui][uj] = s

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
            friends = self.between[u].items() 
            friends.sort(key=lambda x: x[1], reverse=True)
            self._neighbors[u] = [f for f, s in friends[: self.num_neighbor]]
        t2  = time.time()
        log.debug("neighbors N: %i time: %.2fs" % (self.num_neighbor, t2 - t1))
        return self._neighbors 

    def predict(self, user, item):
        if self._neighbors is None:
            raise Exception("please call neighbors() method before predict().")
        if user in self.checkins:
            if item in self.checkins[user]:
                return self.checkins[user][item]

        s = 0.0
        w = 0.0
        for u in self._neighbors[user]:
            w += self.between[user][u]
            if item in self.checkins[u]:
                s += self.between[user][u]
        if w > 0.0:
            return s / w
        return 0.0

