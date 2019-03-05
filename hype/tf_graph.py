#!/usr/bin/env python3

import numpy as np
import pandas
import h5py
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError


def load_adjacency_matrix(path, format="hdf5", symmetrize=False):
    if format == "hdf5":
        with h5py.File(path, "r") as hf:
            return {
                "ids": hf["ids"].value.astype("int"),
                "neighbors": hf["neighbors"].value.astype("int"),
                "offsets": hf["offsets"].value.astype("int"),
                "weights": hf["weights"].value.astype("float"),
                "objects": hf["objects"].value,
            }
    elif format == "csv":
        df = pandas.read_csv(path, usecols=["id1", "id2", "weight"], engine="c")
        if symmetrize:
            rev = df.copy().rename(columns={"id1": "id2", "id2": "id1"})
            df = pandas.concat([df, rev])

        idmap = {}
        idlist = []

        def convert(id):
            if id not in idmap:
                idmap[id] = len(idlist)
                idlist.append(id)
            return idmap[id]

        df.loc[:, "id1"] = df["id1"].apply(convert)
        df.loc[:, "id2"] = df["id2"].apply(convert)

        groups = df.groupby("id1").apply(lambda x: x.sort_values(by="id2"))
        counts = df.groupby("id1").id2.size()

        ids = groups.index.levels[0].values
        offsets = counts.loc[ids].values
        offsets[1:] = np.cumsum(offsets)[:-1]
        offsets[0] = 0
        neighbors = groups["id2"].values
        weights = groups["weight"].values
        return {
            "ids": ids.astype("int"),
            "offsets": offsets.astype("int"),
            "neighbors": neighbors.astype("int"),
            "weights": weights.astype("float"),
            "objects": np.array(idlist),
        }
    else:
        raise RuntimeError(f"Unsupported file format {format}")


def load_edge_list(path, symmetrize=False):
    df = pandas.read_csv(path, usecols=["id1", "id2", "weight"], engine="c")
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={"id1": "id2", "id2": "id1"})
        df = pandas.concat([df, rev])
    idx, objects = pandas.factorize(df[["id1", "id2"]].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype("int")
    weights = df.weight.values.astype("float")
    return idx, objects.tolist(), weights


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
