# -*- coding: utf-8 -*-

import math
import numpy as np
from .loader import tomatrix
from .models import Recommender

__all__ = ["distance", "KDE", "KDEModel"]

def distance(point_x, point_y):  
    """distance between two point, unit is meter
        usage:
        >>> print distance((1.0, 0.0), (0.0, 0.0)) 
        111319.490793
    """
    EARTH_RADIUS = 6378137.0 
    radlat1 = point_x[0] * math.pi / 180.0  
    radlat2 = point_y[0] * math.pi / 180.0 

    x = radlat1 - radlat2  
    y = (point_x[1] - point_y[1]) * math.pi / 180.0
    c = math.sqrt((math.sin(x * 0.5) ** 2) + 
        math.cos(radlat1) * math.cos(radlat2)* (math.sin(y * 0.5) ** 2))
    lenght = 2.0 * math.asin(c) * EARTH_RADIUS  
    return math.fabs(lenght)


class KDE(object):
    """Estamate problity that a user show up in a poi, using KDE method
       usage:
        >>> cks = {0:[0]}
        >>> locations = {0:(0.0, 0.0), 1: (0.00014, 0.0), 2: (0.00028, 0.0)}
        >>> k = KDE(cks, locations)
        >>> print k.probility(0, 1) > k.probility(0, 2)
        True
        >>> print k.probility(0, 1) 
        0.398893835041
    """
    def __init__(self, checkins, locations, smooth=1.0):
        """
        matrix : user checkin data sparse matrix
        locations: poi latitude and longitude
                   {"loc1": (20.0, 30.0), ...}
        """
        self.matrix = tomatrix(checkins)
        self.locations = locations
        if smooth <= 0.0:
            raise ValueError("smooth should > 0.0")
        self.smooth = smooth

    def probility(self, user, item):
        pois = set(np.nonzero(self.matrix[user])[1])
        if len(pois) == 0 or item in pois:
            return 1.0

        sum_prob = 0.0
        for poi in pois:
            loc_x = self.locations[poi]
            loc_y = self.locations[item]
            _dis = distance(loc_x, loc_y) / 1000.0 # to kilometer
            x = _dis / self.smooth
            prob = math.pow(math.e, -0.5 * math.pow(x, 2))
            sum_prob += prob
        return sum_prob / (math.sqrt(2.0 * math.pi) * self.smooth * len(pois))


class KDEModel(Recommender):
    def __init__(self, checkins, locations, smooth=1.0):
        super(KDEModel, self).__init__(checkins)
        self.kde = KDE(checkins, locations, smooth)

    def predict(self, user, item):
        return self.kde.probility(user, item)

