# -*- coding: utf-8 -*-
import time
import logging
import math

import numpy as np
from .userbase import UserBase, similarity

__all__ = ["TopicKnn"]

log = logging.getLogger(__name__)

class TopicKnn(UserBase):
    """User base K nearest neighbors algorithm.
    """
    def __init__(self, checkins, topics, num_neighbors=10):
        super(TopicKnn, self).__init__(checkins, num_neighbors);
        self.topics = topics

    def __repr__(self):
        return "<TopicKnn [K=%i]>" % self.num_neighbors

    def similarity(self):
        t0 = time.time()
        for ui in xrange(self.num_users):
            for uj in xrange(ui + 1, self.num_users):
                s = similarity(self.topics[ui], self.topics[uj])
                self.between[ui, uj] = s

            if ui % 200 == 0:
                t1 = time.time()
                log.debug("similarity user: %i(%.f%%) time: %.2fs" % \
                        (ui, ui * 100.0 / self.num_users, t1 - t0))


