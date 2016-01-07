# -*- coding: utf-8 -*-

import logging
import numpy as np
import random

log = logging.getLogger(__name__)

def nonzero(matrix, row):
    return set(np.nonzero(matrix[row])[1])


def randint(low, height=None): 
    """
       if height is none, then range is [0, low) 
       else range is [low, height)
    """
    if height:
        return random.randint(low, height - 1)
    else:
        return random.randint(0, low - 1)
