# -*- coding: utf-8 -*-

import time
import cPickle
import logging
import numpy as np
import scipy.sparse as sparse

log = logging.getLogger(__name__)

__all__ = ["load_checkins", "tomatrix", "load_locations"]

def load_checkins(infile, index=None):
    """Load checkins data from file.
       infile: an open file object for read, should support `read` method.
       index: specify the (uid, iid) or (uid, iid, freq) index,
                ie. index=(0,1) or index=(0,1,-1)
       usage:
        >>> import StringIO
        >>> s = StringIO.StringIO("0 1 2\\n0 2 3\\n2 2 4\\n")
        >>> print load_checkins(s, index=[0,1,2])
        {0: [(1, 2), (2, 3)], 2: [(2, 4)]}
    """
    if index is None:
        index = [0, 1]
    t0 = time.time()

    count = 0
    users = set()
    items = set()
    pairs  = set()
    
    checkins = {}
    for line in infile:
        params = line.strip().split()
        user = int(params[index[0]])
        item = int(params[index[1]])
        if len(index) >= 3:
            freq = int(params[index[2]])
        else:
            freq = 1

        if (user, item) in pairs:
            continue
        pairs.add((user, item))

        count += 1
        users.add(user)
        items.add(item)
        if user not in checkins:
            checkins[user] = []
        checkins[user].append((item, freq))

    t1 = time.time()
    log.debug("load %i pairs, %i users, %i pois." % (count, len(users), len(items)))
    log.debug('time %.4f seconds' % (t1 - t0))
    return checkins 


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


def load_locations(infile, index=None):
    """Load locations from input file.
        infile: file pointer, an file kind object.
        index: (loc, lantitude, longititude) index in line,
            default index=(0,1,2)
        usage:
        >>> import StringIO
        >>> s = StringIO.StringIO("0 1.0 2.0\\n1 1.0 3.0\\n")
        >>> print load_locations(s)
        {0: (1.0, 2.0), 1: (1.0, 3.0)}
    """
    if index is None:
        index = (0, 1, 2)
    locations = {}
    for line in infile:
        params = line.strip().split()
        item = int(params[index[0]])
        lat = float(params[index[1]])
        lon = float(params[index[2]])
        locations[item] = (lat, lon)
    return locations
 
