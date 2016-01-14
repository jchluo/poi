# -*- coding: utf-8 -*-

import time
import logging

log = logging.getLogger(__name__)

__all__ = ["load_checkins", "load_locations"]

def load_checkins(infile, index=None, repeat=True):
    """Load checkins data from file.
    infile : an open file object for read, should support `read` method.
    index  : specify the (uid, iid) or (uid, iid, freq) index, if tuple, 
            frequence of each record will set as 1, 
            ie. index=(0,1) default or index=(0,1,-1)
    repeat: if True(default), duplicate record will not be abandoned, 
            frequence value will sum together.
    usage:
     >>> import StringIO
     >>> s = StringIO.StringIO("0 1 2\\n0 2 3\\n2 2 4\\n")
     >>> print load_checkins(s, index=[0,1,2])
     {0: [(1, 2), (2, 3)], 2: [(2, 4)]}
     >>> s = StringIO.StringIO("0 1 2\\n0 1 3\\n")
     >>> print load_checkins(s, index=[0,1,2])
     {0: [(1, 5)]}
    """
    if index is None:
        index = (0, 1) 
    t0 = time.time()

    count = 0
    users = set()
    items = set()
    
    counts = {}
    for line in infile:
        params = line.strip().split()
        user = int(params[index[0]])
        item = int(params[index[1]])
        if len(index) >= 3:
            freq = int(params[index[2]])
        else:
            freq = 1

        count += 1
        users.add(user)
        items.add(item)
        if user not in counts:
            counts[user] = {} 
        if repeat:
            counts[user][item] = counts[user].get(item, 0) + freq
        else:
            counts[user][item] = freq

    checkins = {}
    for user in counts:
        checkins[user] = []
        for item, freq in counts[user].items():
            checkins[user].append((item, freq))
    t1 = time.time()
    log.debug("load %i checkins, %i users, %i pois." % (count, len(users), len(items)))
    log.debug('time %.4f seconds' % (t1 - t0))
    return checkins 


def format_checkins(checkins):
    """Make checkins. 
    checkins: {uid: [(iid, freq), ...], ...} or {uid: [uid, ...], ...},
             see `load_checkins` for detail.
    return: number of item, num of user, checkins in dict
    usage:
     >>> cks = {0: [1, 2], 1: [1]} 
     >>> print format_checkins(cks) 
     (2, 3, {0: {1: 1.0, 2: 1.0}, 1: {1: 1.0}})
    """
    counts = {}
    users = set()
    items = set()
    
    for user in checkins:
        users.add(user)
        feedbacks = checkins[user] 
        counts[user] = {}
        for feed in feedbacks:
            if type(feed) == int:
                item = feed
                freq = 1.0
            else:
                item = feed[0]
                freq = feed[1]

            items.add(item)
            counts[user][item] = float(freq)
        
    nuser = max(users) + 1
    nitem = max(items) + 1 
    # add empty {}
    for user in xrange(nuser):
        if user not in counts:
            counts[user] = {} 

    return (nuser, nitem, counts)


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
 
