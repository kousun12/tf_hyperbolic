#!/usr/bin/env python3

import matplotlib.pyplot as plt
import re
import umap
import numpy as np
from collections import namedtuple
import tensorflow as tf

tf.enable_eager_execution()

from embed import MANIFOLDS
from hype.sn import initialize
from hype.graph import load_edge_list

plt.style.use("ggplot")

# A basic viz helper to plot embedding samples over a disk


def poincare_dist(u, v):
    max = 1 - 1e-5
    sq_u_norm = np.clip(np.sum(u * u), a_min=0, a_max=max)
    sq_v_norm = np.clip(np.sum(v * v), a_min=0, a_max=max)
    sq_dist = np.sum(np.power(u - v, 2))
    return np.arccosh(1 + (sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm))) * 2)


def umap_plot(names, embeddings, name, dim, take=None):
    umap_e = umap.UMAP(metric=poincare_dist).fit_transform(embeddings)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()
    for i, w in enumerate(names[:take]):
        x, y = umap_e[i]
        ax.plot(x, y, "o", color="r")
        ax.text(x - 0.1, y + 0.04, re.sub("\.n\.\d{2}", "", w), color="b")
    fig.savefig("plots/" + name + "-umap.png", dpi=fig.dpi)
    plt.show()


def poincare_plot(names, embeddings, name, take=100):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.add_artist(plt.Circle((0, 0), 1.0, color="black", fill=False))
    for i, w in enumerate(names[:take]):
        x, y, *rest = embeddings[i]
        ax.plot(x, y, "o", color="r")
        ax.text(x - 0.1, y + 0.04, re.sub("\.n\.\d{2}", "", w), color="b")
    fig.savefig("plots/" + name + ".png", dpi=fig.dpi)
    plt.show()


Opts = namedtuple("Opts", "manifold dim negs batchsize burnin dampening")
opt = Opts("poincare", 5, 50, 50, 20, 0.75)
closure_name = "wordnet/mammal_closure.csv"
ck_name = "mammals-5d.tf"

if __name__ == "__main__":
    manifold = MANIFOLDS[opt.manifold](debug=False, max_norm=500_000)
    idx, objects, weights = load_edge_list(closure_name, False)
    model, data, model_name = initialize(
        manifold,
        idx,
        objects,
        weights,
        sparse=False,
        dim=opt.dim,
        negs=opt.negs,
        batch_size=opt.batchsize,
        burnin=opt.burnin,
        dampening=opt.dampening,
    )
    model.load_weights(f"checkpoints/{ck_name}")
    poincare_plot(objects, model.emb.numpy(), ck_name, take=80)
    # umap_plot(objects, model.emb.numpy(), ck_name, opt.dim, take=80)
