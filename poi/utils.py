# -*- coding: utf-8 -*-

import logging
import random
from multiprocessing import Pool 

import numpy as np
try:
    import scipy.sparse as sparse
except:
    pass

log = logging.getLogger(__name__)

def nonzero(matrix, row):
    """
    usage:
     >>> from scipy.sparse import csr_matrix
     >>> m = csr_matrix([[1, 0, 2], [3, 0, 0]])
     >>> print nonzero(m, 0)
     set([0, 2])
     >>> print nonzero(m, 1)
     set([0])
    """
    return set(matrix[row].indices)


def randint(low, height=None): 
    """
       if height is none, then range is [0, low) 
       else range is [low, height)
       usage:
        >>> n = randint(0, 10)
        >>> print 0 <= n < 10 
        True
        >>> n = randint(3)
        >>> print 0 <= n < 3
        True
    """
    if height:
        return random.randint(low, height - 1)
    else:
        return random.randint(0, low - 1)


def threads(func, params, num=4, output=True):
    pool = Pool(num)
    if output:
        results = pool.map(func, params) 
    else:
        pool.map(func, params) 
    pool.close()
    pool.join()
    if output:
        return results


def linspace(low, height, num=4): 
    """[low, height)
    usage:
     >>> print linspace(1, 6, 4)
     [(1, 2), (2, 3), (3, 4), (4, 6)]
    """
    lenght = height - low
    size = lenght / num 
    spaces = []
    for i in xrange(num - 1):
        spaces.append((low + i * size, low + (i + 1) * size))

    if lenght % num == 0:
        spaces.append((low + (num - 1) * size, low + num * size ))
    else:
        spaces.append((low + (num - 1) * size, low + num * size + 1))

    return spaces
    


def tomatrix(checkins):
    """Make checkins to a `sparse matrix` object.
    checkins: {uid: [(iid, freq), ...], ...} or {uid: [uid, ...], ...},
             see `load_checkins` for detail.
    usage:
     >>> import StringIO
     >>> s = StringIO.StringIO("0 1 2\\n0 2 3\\n2 2 4\\n")
     >>> cks = load_checkins(s, index=[0,1,2])
     >>> m = tomatrix(cks)
     >>> print ("%s" % m).replace("\\t", " ")
       (0, 1) 2
       (0, 2) 3
       (2, 2) 4
    """
    row = []
    col = []
    data = []

    users = set()
    items = set()
    
    for user in checkins:
        feedbacks = checkins[user] 
        for feed in feedbacks:
            if type(feed) == int:
                item = feed
                freq = 1
            else:
                item = feed[0]
                freq = feed[1]
            row.append(user)
            col.append(item)
            data.append(freq)

            items.add(item)
        
        users.add(user)

    matrix = sparse.csr_matrix((data, (row, col)), shape=(max(users) + 1, max(items) + 1))
    return matrix


