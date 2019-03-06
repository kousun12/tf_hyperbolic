#!/usr/bin/env python3

import tensorflow as tf

tf.enable_eager_execution()
import sys
import json
import logging
import argparse
import numpy as np

from tensorflow.python.framework import ops
from hype.train import train
from hype.sn import initialize
from hype.adjacency_matrix_dataset import AdjacencyDataset
from hype.graph import load_adjacency_matrix, load_edge_list
from hype.embedding import Embedding
from hype.rsgd import RSGDTF
from hype.euclidean import EuclideanManifold
from hype.poincare import PoincareManifold

tf.random.set_random_seed(42)
np.random.seed(42)

MANIFOLDS = {"euclidean": EuclideanManifold, "poincare": PoincareManifold}
DEFAULT_CHPT = "/tmp/hype_embeddings.pth"


def main():
    opt = parse_args()
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger(opt.manifold)
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stdout)

    manifold = MANIFOLDS[opt.manifold](debug=opt.debug, max_norm=opt.maxnorm)
    opt.dim = manifold.dim(opt.dim)

    model, data = _load(opt, log, manifold)
    _run(model, data, opt, log)
    print(model.summary())


def _run(model, data, opt, log):
    epochs = range(opt.epochs)
    lr_base = ops.convert_to_tensor(opt.lr, name="learning_rate", dtype=tf.float64)

    for epoch in epochs:
        data.burnin = opt.burnin > 0 and epoch < opt.burnin
        lr_mult = (
            tf.constant(opt.burnin_multiplier, dtype=tf.float64)
            if data.burnin
            else tf.constant(1, dtype=tf.float64)
        )
        lr = lr_base * lr_mult
        e_losses = tf.constant([], dtype=tf.float64)
        for batch, (inputs, outputs) in enumerate(data):
            cur_loss = train(model, inputs, outputs, learning_rate=lr)
            if cur_loss is not None:
                e_losses = tf.concat([e_losses, [cur_loss]], axis=0)

        if epoch % opt.eval_each == 0:
            log.info(f"epoch {epoch} - loss: {tf.reduce_mean(e_losses)}, lr: {lr}]")

        model.save_weights(opt.checkpoint)  # TODO - use model.save()


def _load(opt, log, manifold):
    if "csv" in opt.dset:
        log.info("Using edge list dataloader")
        idx, objects, weights = load_edge_list(opt.dset, opt.sym)
        model, data, model_name = initialize(
            manifold,
            idx,
            objects,
            weights,
            sparse=opt.sparse,
            dim=opt.dim,
            negs=opt.negs,
            batch_size=opt.batchsize,
            burnin=opt.burnin,
            dampening=opt.dampening,
        )
    else:
        log.info("Using adjacency matrix dataloader")
        dset = load_adjacency_matrix(opt.dset, "hdf5")
        # noinspection PyArgumentList
        data = AdjacencyDataset(
            dset,
            opt.negs,
            opt.batchsize,
            burnin=opt.burnin > 0,
            sample_dampening=opt.dampening,
        )
        model = Embedding(data.N, opt.dim, manifold, sparse=opt.sparse)

    data.neg_multiplier = opt.neg_multiplier
    log.info(f"conf: {json.dumps(vars(opt))}")
    if opt.checkpoint and not opt.fresh:
        log.info("using loaded checkpoint")
        try:
            model.load_weights(opt.checkpoint)
        except Exception:
            log.info(f"not loading existing weights for {opt.checkpoint}")
    else:
        log.info("starting with fresh model")
    return model, data


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hyperbolic Embeddings")
    parser.add_argument(
        "-checkpoint", default=DEFAULT_CHPT, help="Where to store the model checkpoint"
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
    parser.add_argument("-batchsize", type=int, default=100, help="Batchsize")
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
        "-gpu", default=-1, type=int, help="Which GPU to run on (-1 for no gpu)"
    )
    parser.add_argument(
        "-sym", action="store_true", default=False, help="Symmetrize dataset"
    )
    parser.add_argument("-maxnorm", default="500000", type=int)
    parser.add_argument(
        "-sparse",
        default=False,
        action="store_true",
        help="Use sparse gradients for embedding table",
    )
    parser.add_argument("-burnin_multiplier", default=0.01, type=float)
    parser.add_argument("-neg_multiplier", default=1.0, type=float)
    opt = parser.parse_args()
    if opt.gpu >= 0 and not tf.test.is_gpu_available():
        opt.gpu = -1
        print(f"no gpu, defaulting to CPU...")

    return opt


if __name__ == "__main__":
    main()
