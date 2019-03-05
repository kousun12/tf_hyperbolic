#!/usr/bin/env python3

import tensorflow as tf

from plot import poincare_plot

tf.enable_eager_execution()
import math
import sys
import json
import ntpath
import logging
import argparse
import numpy as np

from tensorflow.python.framework import ops
from hype.tf_graph import train
from hype.sn import initialize
from hype.adjacency_matrix_dataset import AdjacencyDataset
from hype.tf_graph import load_adjacency_matrix, load_edge_list
from hype.embedding import Embedding
from hype.rsgd import RSGDTF
from hype.euclidean import EuclideanManifold
from hype.poincare import PoincareManifold

tf.random.set_random_seed(42)
np.random.seed(42)

MANIFOLDS = {"euclidean": EuclideanManifold, "poincare": PoincareManifold}


class Unsettable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(Unsettable, self).__init__(option_strings, dest, nargs="?", **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        val = None if option_string.startswith("-no") else values
        setattr(namespace, self.dest, val)


def main():
    parser = argparse.ArgumentParser(description="Train Hyperbolic Embeddings")
    parser.add_argument(
        "-checkpoint",
        default="/tmp/hype_embeddings.pth",
        help="Where to store the model checkpoint",
    )
    parser.add_argument("-dset", type=str, required=True, help="Dataset identifier")
    parser.add_argument("-dim", type=int, default=20, help="Embedding dimension")
    parser.add_argument(
        "-manifold",
        type=str,
        default="poincare",
        choices=MANIFOLDS.keys(),
        help="Embedding manifold",
    )
    parser.add_argument("-lr", type=float, default=1000, help="Learning rate")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-batchsize", type=int, default=12800, help="Batchsize")
    parser.add_argument("-negs", type=int, default=50, help="Number of negatives")
    parser.add_argument("-burnin", type=int, default=20, help="Epochs of burn in")
    parser.add_argument(
        "-dampening", type=float, default=0.75, help="Sample dampening during burnin"
    )
    parser.add_argument(
        "-eval_each", type=int, default=1, help="Run evaluation every n-th epoch"
    )
    parser.add_argument(
        "-fresh", action="store_true", default=False, help="Override checkpoint"
    )
    parser.add_argument(
        "-debug", action="store_true", default=False, help="Print debugging output"
    )
    parser.add_argument(
        "-gpu", default=0, type=int, help="Which GPU to run on (-1 for no gpu)"
    )
    parser.add_argument(
        "-sym", action="store_true", default=False, help="Symmetrize dataset"
    )
    parser.add_argument(
        "-maxnorm", "-no-maxnorm", default="500000", action=Unsettable, type=int
    )
    parser.add_argument(
        "-sparse",
        default=False,
        action="store_true",
        help="Use sparse gradients for embedding table",
    )
    parser.add_argument("-burnin_multiplier", default=0.01, type=float)
    parser.add_argument("-neg_multiplier", default=1.0, type=float)
    parser.add_argument("-quiet", action="store_true", default=False)
    parser.add_argument(
        "-train_threads",
        type=int,
        default=1,
        help="Number of threads to use in training",
    )
    opt = parser.parse_args()

    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger(opt.manifold)
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stdout)

    if opt.gpu >= 0 and opt.train_threads > 1:
        opt.gpu = -1
        log.warning(f"defaulting to CPU...")

    manifold = MANIFOLDS[opt.manifold](debug=opt.debug, max_norm=opt.maxnorm)
    opt.dim = manifold.dim(opt.dim)

    if "csv" in opt.dset:
        log.info("Using edge list dataloader")
        idx, objects, weights = load_edge_list(opt.dset, opt.sym)
        model, data, model_name, conf = initialize(
            manifold, opt, idx, objects, weights, sparse=opt.sparse
        )
    else:
        log.info("Using adjacency matrix dataloader")
        dset = load_adjacency_matrix(opt.dset, "hdf5")
        log.info("Setting up dataset...")
        # noinspection PyArgumentList
        data = AdjacencyDataset(
            dset,
            opt.negs,
            opt.batchsize,
            burnin=opt.burnin > 0,
            sample_dampening=opt.dampening,
        )
        model = Embedding(data.N, opt.dim, manifold, sparse=opt.sparse)

    # set burnin parameters
    data.neg_multiplier = opt.neg_multiplier
    train._lr_multiplier = opt.burnin_multiplier
    log.info(f"json_conf: {json.dumps(vars(opt))}")

    # optimizer = RSGDTF(learning_rate=opt.lr, rgrad=manifold.rgrad, expm=manifold.expm)
    # model.compile(optimizer=optimizer,
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    def exp_decay(step):
        initial_rate = opt.lr * 1.75
        k = 0.05
        return max(initial_rate * math.exp(-k * step), opt.lr)

    if opt.checkpoint and not opt.fresh:
        print("using loaded checkpoint")
        try:
            model.load_weights(opt.checkpoint)
        except:
            print(f"can not load existing weights for {opt.checkpoint}")
    else:
        print("starting with fresh model")

    with tf.device("/cpu:0"):
        num_epochs = 120
        epochs = range(num_epochs)
        lr = ops.convert_to_tensor(opt.lr, name="learning_rate", dtype=tf.float64)
        for epoch in epochs:
            losses = tf.constant([], dtype=tf.float64)
            for batch, (inputs, outputs) in enumerate(data):
                cur_loss = train(model, inputs, outputs, learning_rate=lr)
                losses = tf.concat([losses, [cur_loss]], axis=0)

            print(f"epoch {epoch} - loss: {tf.reduce_mean(losses)}, lr: {lr}]")
            # TODO - use model.save()
            checkpoint_path = opt.checkpoint or "/tmp/hype_emb.tf"
            model.save_weights(checkpoint_path)
        print(model.summary())
        poincare_plot(objects, model.emb.numpy(), ntpath.basename(checkpoint_path))


if __name__ == "__main__":
    main()
