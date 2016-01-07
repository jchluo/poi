# -*- coding: utf-8 -*-

import time
import logging
from multiprocessing import Pool 
import numpy as np

from .loader import tomatrix

log = logging.getLogger(__name__)

__all__ = ["Recommender", "Evaluation"]

class Recommender(object):
    def __init__(self, checkins):
        super(Recommender, self).__init__()
        self.checkins = checkins
        self.matrix = tomatrix(checkins)
        self.num_users = self.matrix.shape[0]
        self.num_items = self.matrix.shape[1]

    def train(self, before=None, after=None):
        raise NotImplementedError

    def predict(self, user, item):
        raise NotImplementedError

    def recommend(self, user, num=5, ruleout=True):
        scores = []
        for poi in xrange(self.num_items):
            scores.append((poi, self.predict(user, poi)))
        scores.sort(key=lambda x: x[1], reverse=True)

        if self.matrix is not None and ruleout:
            ruleouts = set(np.nonzero(self.matrix[user])[1])
        else:
            ruleouts = set()

        result = []
        for poi, score in scores:
            if poi in ruleouts:
                continue
            result.append(poi)
            if len(result) >= num:
                break
        return result 

        
def _proxy_test(args):
    evaluation, user, full = args
    bingos = evaluation.hits(user)
    n = len(bingos)
    if full and n > 0:
        log.debug("user %i hit %s" % (user, bingos))
    return (user, n)


class Evaluation(object):
    def __init__(self, 
                checkins, 
                model=None, 
                topN=5, 
                users=None, 
                _pool_num=4, 
                full=True):
        """
        Evaluate a model.Report precision and recall.
        checkins: test checkins, set `loader.load_checkins` method for more informations 
        model: model for test, must has `recommend` methid
        N    : recommend N pois
        users: users for test, should be iterated
        _pool_num: thread number to test, most cases default is ok.
                    if 0, then turn off multiple threads.
        full: log hit record to screen and file, default True
        usage:
        >>> cks = {0: [1], 1:[0, 1], 2:[1,2]}
        >>> class M(object):
        ...     def recommend(self, u, N):
        ...         if u == 0:
        ...             return [1]
        ...         return []
        >>> ev = Evaluation(cks, model=M(), users=[0, 1], _pool_num=0)
        >>> ev.assess()
        (0.5, 0.1)

        """
        self.matrix = tomatrix(checkins)
        self.topN = topN
        self.model = model
        self._pool_num = _pool_num
        self.num_users = self.matrix.shape[0]
        self.num_items = self.matrix.shape[1]
        if users is None:
            self.users = xrange(self.num_users)
        else:
            self.users = users

    def hits(self, user):
        pois = set(np.nonzero(self.matrix[user])[1])
        if len(pois) <= 0:
            return []
        result = self.model.recommend(user, self.topN)
        return list(set(pois) & set(result))

    def assess(self, model=None, topN=None, users=None, full=None):
        if model is not None:
            self.model = model
        if self.model is None:
            raise ValueError("model is None.")
        if topN is not None:
            self.topN = topN
        if users is not None:
            self.users = users
        if full is not None:
            self.full = full
        
        t0 = time.time()
        def prepare():
            for user in self.users:
                yield (self, user, full)

        if self._pool_num > 0:
            pool = Pool(self._pool_num)
            matchs = pool.map(_proxy_test, prepare()) 
            pool.close()
            pool.join()
        else:
            matchs = []
            for arg in prepare():
                matchs.append(_proxy_test(arg))
        
        nhits = sum([n for u, n in matchs])
        _recall = 0.0
        valid_num = 0
        for user, n in matchs:
            pois = np.nonzero(self.matrix[user])[1]
            if len(pois) > 0:
                valid_num += 1
                _recall += float(n) / len(pois)

        if valid_num == 0:
            raise ValueError("Checkin matrix should not be empty.")
        prec = float(nhits) / (valid_num * self.topN)
        _recall = float(_recall) / valid_num 
        t1 = time.time()
        log.info("recall   : %.4f" % _recall)
        log.info("precision: %.4f" % prec)
        log.debug('time %.4f seconds' % (t1 - t0))
        return (_recall, prec)

