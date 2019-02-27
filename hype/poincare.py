#!/usr/bin/env python3

import tensorflow as tf
from .euclidean import EuclideanManifold


class PoincareManifold(EuclideanManifold):
    def __init__(self, eps=1e-14, **kwargs):
        super(PoincareManifold, self).__init__(**kwargs)
        self.eps = eps
        self.boundary = 1 - eps
        self.max_norm = self.boundary

    def distance(self, u, v):
        sq_u_norm = tf.clip_by_value(
            tf.reduce_sum(u * u, axis=-1), clip_value_min=0, clip_value_max=1 - self.eps
        )
        sq_v_norm = tf.clip_by_value(
            tf.reduce_sum(v * v, axis=-1), clip_value_min=0, clip_value_max=1 - self.eps
        )
        sq_dist = tf.reduce_sum(tf.pow(u - v, 2), axis=-1)
        x = sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm)) * 2 + 1
        z = tf.sqrt(tf.pow(x, 2) - 1)
        return tf.log(x + z)

    def rgrad(self, p, d_p):
        return d_p * ((1 - tf.reduce_sum(p ** 2)) ** 2 / 4)
