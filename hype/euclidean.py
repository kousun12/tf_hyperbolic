#!/usr/bin/env python3
from abc import ABC

from .manifold import Manifold
import tensorflow as tf


class EuclideanManifold(Manifold, ABC):
    __slots__ = ["max_norm"]
    name = "euclidean"

    def __init__(self, max_norm=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def normalize(self, u):
        d = u.shape[-1].value
        return tf.clip_by_norm(tf.reshape(u, [-1, d]), self.max_norm, axes=0)

    def distance(self, u, v):
        return tf.reduce_sum(tf.pow((u - v), 2))

    def pnorm(self, u, dim=None):
        return tf.sqrt(tf.reduce_sum(u * u, axis=dim))

    def rgrad(self, p, d_p):
        return d_p

    def expm(self, p, d_p, normalize=False, lr=None, out=None):
        if lr is not None:
            d_p = d_p * -lr
        if out is None:
            out = p
        out = out + d_p
        if normalize:
            self.normalize(out)
        return out

    def logm(self, p, d_p, out=None):
        return p - d_p
