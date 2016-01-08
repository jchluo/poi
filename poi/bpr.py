# -*- coding: utf-8 -*-

import time
import logging
import math

import numpy as np
from .models import Recommender 
from .utils import nonzero
from .utils import randint
from .utils import threads 

__all__ = ["BPR"]

log = logging.getLogger(__name__)


def _proxy_samples(args):
    model, size = args
    return model.create_samples(size)


class BPR(Recommender):
    def __init__(self, 
                 checkins, 
                 num_factors=10, 
                 num_iterations=100,
                 learn_rate=0.1,
                 decay_rate=0.999,
                 reg_user=0.02,
                 reg_item=0.02,
                 reg_bias=0.02,
                 size_batch=None):
        super(BPR, self).__init__(checkins);
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.reg_bias = reg_bias
        self.current = 0 
        self.num_batchs = 500 # decay learn rate how many times in a iteration
        if size_batch is None:
            self.size_batch = int(math.sqrt(self.num_users) * 100);
        else:
            self.size_batch = size_batch 
            
        # INIT should be small float
        self.user_vectors = 0.1 * np.random.normal(
                            size=(self.num_users, self.num_factors))
        self.item_vectors = 0.1 * np.random.normal(
                            size=(self.num_items, self.num_factors))

        self._nonzero = {}
        for i in xrange(self.num_users):
            self._nonzero[i] = nonzero(self.matrix, i)

    def create_samples(self, size):
        samples = []
        for i in xrange(size):
            locs = set()
            while len(locs) == 0:
                rand_user = randint(self.num_users)
                locs = self._nonzero[rand_user]

            rand_item = randint(self.num_items)
            if rand_item in locs:
                neg_item = randint(self.num_items)
                while neg_item in locs:
                    neg_item = randint(self.num_items)
                wrap = (rand_user, rand_item, neg_item) 
                samples.append(wrap)
            else:
                locs = list(locs)
                pos_item = locs[randint(len(locs))]
                wrap = (rand_user, pos_item, rand_item) 
                samples.append(wrap)
        return samples

    def train(self, before=None, after=None):
        while self.current < self.num_iterations:
            t0 = time.time()
            self.current += 1 

            if before is not None:
                before(self)

            # samples
            samples = []

            t3 = time.time()
            # multiple threads
            params = [(self, self.size_batch) for i in xrange(self.num_batchs)]
            samples = threads(_proxy_samples, params)

            t4 = time.time()
            log.debug("sample %i, time %.2f" % (self.num_batchs * self.size_batch, (t4 - t3)))

            # update a batch
            for b in xrange(self.num_batchs):
                # update learn rate
                self.learn_rate *= self.decay_rate

                # udpate 
                for user, pos, neg in samples[b]: 
                    x = self.predict(user, pos) - self.predict(user, neg)
                    z = 1.0 / (1.0 + math.exp(x))
                    uvec = self.user_vectors[user]
                    ipvec = self.item_vectors[pos]
                    invec = self.item_vectors[neg]

                    self.user_vectors[user] += \
                        self.learn_rate * ((ipvec - invec) * z - self.reg_user * uvec)
                    self.item_vectors[pos] += \
                        self.learn_rate * (uvec * z - self.reg_item * ipvec)
                    self.item_vectors[neg] += \
                            self.learn_rate * (-uvec * z - self.reg_item * invec)
                
            t5 = time.time()
            log.debug('update, time %.2f' % (t5 - t4))
            if after is not None:
                after(self)
            t1 = time.time()
            log.debug('Iteration %i finished, time %.2f' % (self.current, t1 - t0))

    def predict(self, user, item):
        return self.user_vectors[user].T.dot(self.item_vectors[item])

