# -*- coding: utf-8 -*-

"""Cache Recommender.
    dump : run topN predict item for each user, and 
        dump them to file like object(disk file or memory).
    load : recover from file like object, return CacheRecommender.
        Note that this recommender just a tiny version of the original one,
        which can only predict topN (stored in file) items.
    usage:
    >>> class M(object):
    ...    def __init__(self):
    ...       self.num_users = 1
    ...       self.num_items = 3
    ...       self.checkins = {0: {0:1}}
    ...       self.name = "Test"
    ...    def predict(self, u, i):
    ...       return 1.0 * i

    usage dump:
    >>> from StringIO import StringIO
    >>> f = StringIO()
    >>> md = M()
    >>> dump(md, f, attrs=["name"], num_pool=0)

    usage load
    >>> f.seek(0) 
    >>> cr = load(f)
    >>> print cr.predict(0, 2)
    2.0
    >>> print cr.name
    Test
"""

import time
import json
import logging

import numpy as np
from .utils import threads
from .models import Recommender


log = logging.getLogger(__name__)

__all__ = ["Recommender", "Evaluation"]

class CacheRecommender(Recommender):
    """Cache File Recommender.
    """
    def __init__(self):
        self.checkins = {}
        self._data = {}
        self._meta = {}

    def __getattr__(self, attr):
        if attr == "_meta":
            raise AttributeError()
        if attr in self._meta:
            return self._meta[attr]
        raise AttributeError("attribute: %s Not Found." % attr)

    def __repr__(self):
        return "<Cache %s>" % self._meta["__repr__"][1: -1]

    def predict(self, user, item):
        return self._data.get(user, {}).get(item, -10 * 10)


def _proxy_predict(arg):
    model, i, num = arg
    scores = [(j, model.predict(i, j)) for j in xrange(model.num_items)\
            if j not in model.checkins[i]]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i, scores[: num]]


def dump(model, fp, num=1000, attrs=None, num_pool=4):
    """Dump predict record to file.
        fp: file pointer like object, 
        num: top num item and its score will be stored,
            other item will be abandoned.
        attrs: list like, the attributes want to be stored,
                num_items and num_users will auto stored.
        num_pool: number of threads, 0 will turn off multiple threads.
    """
    if model is None:
        raise ValueError("model is None.") 

    t0 = time.time()
    args = [(model, i, num) for i in xrange(model.num_users)]
    if num_pool > 0:
        results = threads(_proxy_predict, args, num_pool)
    else:
        results = [_proxy_predict(arg) for arg in args]

    meta = {}
    # write attributes
    if attrs is None:
        attrs = ["num_users", "num_items"]
    else:
        attrs = list(attrs)
        attrs.extend(["num_users", "num_items"])
        attrs = set(attrs)
    for attr in attrs:
        if not hasattr(model, attr):
            raise AttributeError("attribute: %s Not Found." % attr)
        meta[attr] = getattr(model, attr)
    # write __repr__
    meta["__repr__"] = str(model)
    print >> fp, json.dumps(meta)
    # write recoreds
    for one in results: 
        print >> fp, json.dumps(one)

    t1 = time.time()
    log.debug("dump ok, time: %.2fs" % (t1 - t0))


def load(fp):
    """Reture a cacherecommender, which is the tiny version of the 
    original one.
    fp: file like object.
    """
    cr = CacheRecommender()
    # meta
    cr._meta = json.loads(fp.readline())
    # recoreds
    for line in fp:
        rd = json.loads(line.strip())
        user = int(rd[0])
        scores = rd[1]
        cr._data[user] = {}
        for l, s in scores:
            cr._data[user][int(l)] = float(s)
    return cr

