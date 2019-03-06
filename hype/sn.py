#!/usr/bin/env python3

from .graph_dataset import BatchedDataset
from .embedding import Embedding

name_template = "%s_dim%d"


def initialize(
    manifold,
    idx,
    objects,
    weights,
    dim=20,
    negs=50,
    batch_size=100,
    burnin=20,
    dampening=0.75,
    sparse=False,
):
    # noinspection PyArgumentList
    data = BatchedDataset(
        idx,
        objects,
        weights,
        negs,
        batch_size,
        burnin=burnin > 0,
        sample_dampening=dampening,
    )
    # data = Dataset(idx, objects, weights, negs)
    model = Embedding(len(data.objects), dim, manifold, sparse=sparse)
    data.objects = objects
    return model, data, name_template % (manifold.name, dim)
