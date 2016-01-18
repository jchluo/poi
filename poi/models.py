# -*- coding: utf-8 -*-

import time
import logging

import numpy as np

from .loader import format_checkins
from .utils import threads

log = logging.getLogger(__name__)

__all__ = ["Recommender", "Evaluation"]

class Recommender(object):
    """Recommender Base class.
    """
    def __init__(self, checkins=None):
        super(Recommender, self).__init__()
        if checkins is not None:
            self.checkins = checkins
            result = format_checkins(checkins) 
            self.num_users, self.num_items, self.checkins = result 
        else:
            self.checkins = {} 

    def train(self, before=None, after=None):
        raise NotImplementedError

    def predict(self, user, item):
        raise NotImplementedError

    def recommend(self, user, num=5, ruleout=True):
        if user in self.checkins and ruleout:
            ruleouts = set(self.checkins[user].keys())
        else:
            ruleouts = set()

        scores = []
        for poi in xrange(self.num_items):
            if poi in ruleouts:
                continue
            scores.append((poi, self.predict(user, poi)))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [poi for poi, s in scores[: num]] 

        
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
        result = format_checkins(checkins) 
        self.num_users, self.num_items, self.checkins = result 
        self.topN = topN
        self.model = model
        self._pool_num = _pool_num
        self.full = full
        self.precision = 0.0 
        self.recall = 0.0 
        if users is None:
            self.users = xrange(self.num_users)
        else:
            self.users = users

    def __repr__(self):
        return "<Eval [N=%i, prec=%.4f, reca=%.4f]>" %\
                (self.topN, self.precision, self.recall)

    def hits(self, user):
        if user not in self.checkins:
            return []
        pois = set(self.checkins[user].keys())
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

        args = [(self, i, self.full) for i in self.users]
        if self._pool_num > 0:
            matchs = threads(_proxy_test, args, num=self._pool_num)
        else:
            matchs = [_proxy_test(arg) for arg in args]
        
        nhits = sum([n for u, n in matchs])
        reca = 0.0
        valid_num = 0
        for user, n in matchs:
            if user in self.checkins:
                pois = set(self.checkins[user].keys())
            else:
                pois = []
            if len(pois) > 0:
                valid_num += 1
                reca += float(n) / len(pois)

        if valid_num == 0:
            raise ValueError("Checkin matrix should not be empty.")
        prec = float(nhits) / (len(self.users) * self.topN)
        reca = float(reca) / len(self.users) 
        t1 = time.time()
        log.info("recall   : %.4f" % reca)
        log.info("precision: %.4f" % prec)
        log.info('time     : %.4fs' % (t1 - t0))

        self.precision = prec
        self.recall = reca

        return (reca, prec)


def assess(model, checkins, topN=None, users=None, full=None, num_pool=3):
    eva = Evaluation(checkins, model=model, _pool_num=num_pool)
    eva.assess(topN=topN, users=users, full=full)
    return eva
