# -*- coding: utf-8 -*-

import time
import logging
import math

import numpy as np
from .models import Recommender 
from .utils import nonzero
from .utils import randint

__all__ = ["BPR"]

log = logging.getLogger(__name__)

class BPR(Recommender):
    def __init__(self, 
                 checkins, 
                 num_factors=10, 
                 num_iterations=10000,
                 learn_rate=0.1,
                 decay_rate=0.999,
                 reg_user=0.02,
                 reg_item=0.02,
                 reg_bias=0.02,
                 num_samples=None):
        super(BPR, self).__init__(checkins);
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.reg_bias = reg_bias
        self.current = 0 
        if num_samples is None:
            self.num_samples = int(math.sqrt(self.num_users) * 100);
        else:
            self.num_samples = num_samples
            
        print self.num_samples
        # INIT should be small float
        self.user_vectors = 0.1 * np.random.normal(
                            size=(self.num_users, self.num_factors))
        self.item_vectors = 0.1 * np.random.normal(
                            size=(self.num_items, self.num_factors))

    def _create_samples(self):
        for i in xrange(self.num_samples):
            locs = set()
            while len(locs) == 0:
                rand_user = randint(self.num_users)
                locs = nonzero(self.matrix, rand_user) 

            rand_item = randint(self.num_items)
            if rand_item in locs:
                neg_item = randint(self.num_items)
                while neg_item in locs:
                    neg_item = randint(self.num_items)
                yield (rand_user, rand_item, neg_item) 
            else:
                locs = list(locs)
                pos_item = locs[randint(len(locs))]
                yield (rand_user, pos_item, rand_item) 

    def train(self, before=None, after=None):
        t0 = time.time()
        while self.current < self.num_iterations:
            self.current += 1 
            if before is not None:
                before(self)
            # update learn rate
            self.learn_rate *= self.decay_rate
            
            # sample
            samples = self._create_samples() 
            for user, pos, neg in samples: 
                x = self.predict(user, pos) - self.predict(user, neg)
                z = 1.0 / (1.0 + math.exp(x))
                for f in xrange(self.num_factors):
                    uuf = self.user_vectors[user][f]
                    ipf = self.item_vectors[pos][f]
                    inf = self.item_vectors[neg][f]
                    self.user_vectors[user][f] += self.learn_rate * ((ipf - inf) * z - self.reg_user * uuf)
                    self.item_vectors[pos][f] += self.learn_rate * (uuf * z - self.reg_item * ipf)
                    self.item_vectors[neg][f] += self.learn_rate * (-uuf * z - self.reg_item * inf)

            if self.current % 100 == 0:
                t1 = time.time()
                log.debug('iteration %i, time %.2f seconds' % (self.current, t1 - t0))
            if self.current % 500 == 0:
                if after is not None:
                    after(self)
                t1 = time.time()
                log.debug('iteration %i finished in %.2f seconds' % (self.current, t1 - t0))
                t0 = time.time()

    def predict(self, user, item):
        return self.user_vectors[user].T.dot(self.item_vectors[item])

