#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError


def train(model, inputs, outputs, learning_rate=tf.constant(0.3, dtype=tf.float64)):
    with tf.GradientTape() as t:
        t.watch([model.emb])
        try:
            _loss = model.loss(model(inputs), outputs)
        except InvalidArgumentError:
            return None
    d_emb, *rest = t.gradient(_loss, t.watched_variables(), None)
    d_p = model.manifold.rgrad(model.emb, d_emb)
    update = model.manifold.expm(model.emb, d_p, lr=learning_rate)
    model.emb.assign(update)
    return _loss
