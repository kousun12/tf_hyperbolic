#!/usr/bin/env python3

import tensorflow as tf
from .euclidean import EuclideanManifold


class PoincareManifold(EuclideanManifold):
    name = "poincare"

    def __init__(self, eps=1e-14, **kwargs):
        super(PoincareManifold, self).__init__(**kwargs)
        self.eps = eps
        self.max_norm = 1 - eps

    def distance(self, u, v):
        sq_u_norm = tf.clip_by_value(
            tf.reduce_sum(u * u, axis=-1),
            clip_value_min=0,
            clip_value_max=self.max_norm,
        )
        sq_v_norm = tf.clip_by_value(
            tf.reduce_sum(v * v, axis=-1),
            clip_value_min=0,
            clip_value_max=self.max_norm,
        )
        sq_dist = tf.reduce_sum(tf.pow(u - v, 2), axis=-1)
        return tf.acosh(1 + (sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm))) * 2)

    def rgrad(self, p, d_p):
        p_sq_norm = 1 - tf.reduce_sum(tf.pow(p, 2), axis=-1, keepdims=True)
        return d_p * (tf.pow(p_sq_norm, 2) / 4)
