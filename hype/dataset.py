#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from collections import defaultdict as ddict
from numpy.random import randint
from numpy.random import choice


class Dataset(object):
    _neg_multiplier = 1
    _ntries = 10
    _sample_dampening = 0.75

    def __init__(self, idx, objects, weights, nnegs, unigram_size=1e8):
        assert idx.ndim == 2 and idx.shape[1] == 2
        assert weights.ndim == 1
        assert len(idx) == len(weights)
        assert nnegs >= 0
        assert unigram_size >= 0

        print("Indexing data")
        self.idx = idx
        self.nnegs = nnegs
        self.burnin = False
        self.objects = objects

        self._weights = ddict(lambda: ddict(int))
        self._counts = np.ones(len(objects), dtype=np.float)
        self.max_tries = self.nnegs * self._ntries
        for i in range(idx.shape[0]):
            t, h = self.idx[i]
            self._counts[h] += weights[i]
            self._weights[t][h] += weights[i]
        self._weights = dict(self._weights)
        nents = int(np.array(list(self._weights.keys())).max())
        assert len(objects) > nents, f"Number of objects do no match"

        if unigram_size > 0:
            c = self._counts ** self._sample_dampening
            self.unigram_table = choice(
                len(objects), size=int(unigram_size), p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.shape[0]

    def weights(self, inputs, targets):
        return self.fweights(self, inputs, targets)

    def nnegatives(self):
        if self.burnin:
            return self._neg_multiplier * self.nnegs
        else:
            return self.nnegs

    def __getitem__(self, i):
        t, h = self.idx[i]
        negs = set()
        num_tries = 0
        num_negs = int(self.nnegatives())
        if t not in self._weights:
            negs.add(t)
        else:
            while num_tries < self.max_tries and len(negs) < num_negs:
                if self.burnin:
                    n = randint(0, len(self.unigram_table))
                    n = int(self.unigram_table[n])
                else:
                    n = randint(0, len(self.objects))
                if (n not in self._weights[t]) or (
                    self._weights[t][n] < self._weights[t][h]
                ):
                    negs.add(n)
                num_tries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < num_negs + 2:
            ix.append(ix[randint(2, len(ix))])
        return tf.constant(ix, dtype=tf.int64), tf.constant(0, dtype=tf.int64)
