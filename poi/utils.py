# -*- coding: utf-8 -*-

import logging
import numpy as np
import random
from multiprocessing import Pool 

log = logging.getLogger(__name__)

def nonzero(matrix, row):
    return set(matrix[row].nonzero()[1])


def randint(low, height=None): 
    """
       if height is none, then range is [0, low) 
       else range is [low, height)
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
