#!/usr/bin/env python3

import tensorflow as tf


class Embedding(tf.keras.Model):
    def __init__(self, size, dim, manifold, sparse=True):
        super(Embedding, self).__init__()
        self.dim = dim
        self.sparse = sparse
        self.nobjects = size
        self.manifold = manifold
        self.dist = manifold.distance
        self.pre_hook = None
        self.post_hook = None
        self.eps = 1e-10
        scale = 1e-4
        self.emb = tf.Variable(
            tf.random_uniform([size, dim], -scale, scale, dtype=tf.float64), name="emb"
        )

    def _forward(self, e):
        u = tf.strided_slice(e, [0, 0], [e.shape[0], 1])
        v = tf.strided_slice(e, [0, 1], [e.shape[0], e.shape[1]])

        _from = tf.nn.embedding_lookup(self.emb, u)
        _to = tf.nn.embedding_lookup(self.emb, v)
        return -self.dist(_from, _to)

    def loss(self, preds, targets):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=preds)
        )

    def call(self, inputs, training=False, **kwargs):
        # e = self.manifold.normalize(inputs)
        if self.pre_hook is not None:
            inputs = self.pre_hook(inputs)
        return self._forward(inputs)
