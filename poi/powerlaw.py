# -*- coding: utf-8 -*-

import time
import math
import logging

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from .kde import distance

log = logging.getLogger(__name__)

class PowerLaw(object):
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
        self.checkins = checkins
        self.locations = locations
        self.num_loc = len(locations) 
        self.arr_x = []
        self.arr_y = []
        self.points = []
        self.a = 0.0
        self.b = 0.0
        self.line_ready = False

    def prob(self, x):
        """Calculate the probability for x distance.
        y = a * x ^ b
        x : distance, km
        """
        return self.a * np.power(x, self.b)

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
        log.debug("least x: %s" % x)
        log.debug("least y: %s" % y)
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
        far = {}
        for i in range(self.num_loc):
            for j in range(i + 1, self.num_loc):
                pi = self.locations[i]
                pj = self.locations[j]
                d = distance(pi, pj) / 1000
                k = round(d, 1)
                far[(i, j)] = k
                data[k] = 0  
        log.debug("distance cmp ok.")

        for u in self.checkins:
            locs = []
            for en in self.checkins[u]:
                if type(en) in [tuple, list]:
                    locs.append(en[0])
                else:
                    locs.append(en)
            num = len(locs)
            for i in range(num):
                for j in range(i + 1, num):
                    li = locs[i]
                    lj = locs[j]
                    if (li, lj) not in far:
                        li, lj = lj, li
                    k = far[(li, lj)]
                    data[k] += 1
        log.debug("distance count ok.")

        x = []
        y = []
        count = 0
        for k, c in data.items():
            if c > 0:
                x.append(k)
                y.append(float(c))
                count += c
        y = [c / count for c in y]
        arr_x = np.array(x)
        # deal with 0, if x = 0, log(x) will cause an -infinity error.
        #arr_x = np.where(arr_x == 0.0, 10 ** (-20), arr_x)
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
        else:
            plt.show()
        log.debug("plot ok.")

