# -*- coding: utf-8 -*-

import time
import math
import logging

import numpy as np
try:
    import matplotlib.pyplot as plt
    from scipy.optimize import leastsq
except:
    pass

from .kde import distance
from .models import Recommender

log = logging.getLogger(__name__)

def approximate_distance(point_i, point_j):
    d = distance(point_i, point_j) / 1000.0
    k = round(d, 1)
    return k


class PowerLaw(Recommender):
    """Power Law algorithm.
    usage:
     >>> cks = {1:[0, 1, 2, 3], 2:[0, 1, 2]}
     >>> locs = {0:(0.0, 0.0), 1:(0.1,0.1), 2:(0.1, 0.2), 3:(0.0, 0.1)}
     >>> pl = PowerLaw(cks, locs)
     >>> pl.count()
     >>> pl.guass()
     >>> pl.plot("pic.png") 
     >>> round(pl.prob(1.0), 2)
     3.53
    """
    def __init__(self, checkins, locations):
        """Init model.
        checkins: see poi.load_checkins method
        locations: see poi.locations method.
        """
        super(PowerLaw, self).__init__(checkins)
        self.locations = locations
        self.points = []
        self.a = 0.0
        self.b = 0.0
        self.line_ready = False
        self._cache = {}

    def __repr__(self):
        return "<PowerLaw [a=%f, b=%f]>" % (self.a, self.b) 

    def prob(self, x):
        """Calculate the probability for x distance.
        y = a * x ^ b
        x : distance, km
        """
        if x == 0.0:
            #log.warn("Two locations distance is zero.")
            x = 10 ** (-10)
        return self.a * (x ** self.b)

    def predict(self, user, loc):
        """Predict the probability about user will checkin in loc.
        prob = Pr[l|Li] = IIPr[l, li] (li in Li)
        see: Exploiting Geographical Influence for Collaborative 
             Point-of-Interest Recommendation
        """
        if user in self._cache:
            if loc not in self._cache[user]:
                return 0.0
            return self._cache[user][loc]

        self._cache[user] = {}
        max_y = -np.Infinity
        for l in xrange(self.num_items):
            if l in self.checkins[user]:
                continue
            y = 0.0 
            for li in self.checkins[user]:
                d = distance(self.locations[l], self.locations[li]) / 1000.0
                y += np.log(self.prob(d))
            if y > max_y:
                max_y = y
            self._cache[user][l] = y
        for l, y in self._cache[user].items():
            self._cache[user][l] = math.e ** (y - max_y)

        if loc not in self._cache[user]:
            return 0.0
        return self._cache[user][loc]

    def guass(self, max_x=None, min_x=0.0):
        """Run Least Square algorithm to guass the line.
        Only x value in (min_x, max_x] will be use as input points.
        min_x: min x value, lower than min_x will be abandoned,
                default 0.0, when you want to omit left part points,
                assign the value.
        max_y: for omit right part points.
        """
        def _error(w, x, y):
            return (w[0] + w[1] * x) - y
        arr_x = self.points[0]
        arr_y = self.points[1]
        _arr_x = []
        _arr_y = []
        if max_x is not None:
            for i, x in enumerate(arr_x):
                if min_x < x <= max_x:
                    _arr_x.append(x)
                    _arr_y.append(arr_y[i])
        else:
            for i, x in enumerate(arr_x):
                if min_x < x:
                    _arr_x.append(x)
                    _arr_y.append(arr_y[i])
        x = np.log(_arr_x)
        y = np.log(_arr_y) 
        log.debug("least x: [%s, ...]" % str(x[: 4])[1: -1])
        log.debug("least y: [%s, ...]" % str(y[: 4])[1: -1])
        result = leastsq(_error, [1, 1], args=(x, y))
        self.a = np.power(math.e, result[0][0])
        self.b = result[0][1]
        log.debug("least a: %f" % self.a)
        log.debug("least b: %f" % self.b)

        self.line_ready = True

    def count(self):
        """Count distance show up how many times in checkins.
        """
        data = {}
        for u in self.checkins:
            locs = []
            for l, f in self.checkins[u].items():
                locs.append(l)
            num = len(locs)
            for i in range(num):
                for j in range(i + 1, num):
                    pi = self.locations[locs[i]]
                    pj = self.locations[locs[j]]
                    k = approximate_distance(pi, pj) 
                    data[k] = data.get(k, 0) + 1
        x = []
        y = []
        count = 0
        for k, c in data.items():
            x.append(k)
            y.append(float(c))
            count += c
        y = [c / count for c in y]
        arr_x = np.array(x)
        arr_y = np.array(y)
        sort_index = np.argsort(arr_x)
        arr_x = arr_x[sort_index]
        arr_y = arr_y[sort_index]
        self.points = [arr_x, arr_y]

    def plot(self, filename=None, marker='+', color='blue'):
        """Plot Power Law picture.
        filename: if assgin, picture will write to file,
                not show on screen, else show on screen.
        marker  : see http://matplotlib.org/api/markers_api.html
        color   : point color
        """
        plt.loglog(*self.points, linestyle='None', marker=marker, markeredgecolor=color)
        if self.line_ready:
            x = self.points[0]
            x = np.linspace(x[1], max(x), 10)
            y = [self.prob(i) for i in x]
            plt.loglog(x, y, linestyle='-')

        if filename is not None:
            plt.savefig(filename)
            log.debug("plot %s ok." % filename)
        else:
            plt.show()

