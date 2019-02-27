#!/usr/bin/env python3

from .graph_dataset import BatchedDataset
from .embedding import Embedding

model_name = "%s_dim%d"


def initialize(manifold, opt, idx, objects, weights, sparse=False):
    conf = []
    # noinspection PyArgumentList
    data = BatchedDataset(
        idx, objects, weights, opt.negs, opt.batchsize, opt.burnin > 0, opt.dampening
    )
    # data = Dataset(idx, objects, weights, opt.negs)
    model = Embedding(len(data.objects), opt.dim, manifold, sparse=sparse)
    data.objects = objects
    return model, data, model_name % (opt.manifold, opt.dim), conf
