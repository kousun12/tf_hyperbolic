import matplotlib.pyplot as plt
from collections import namedtuple
import re
import tensorflow as tf

tf.enable_eager_execution()

from embed import MANIFOLDS
from hype.sn import initialize
from hype.tf_graph import load_edge_list

plt.style.use("ggplot")


def pplot(names, embeddings, name="mammal"):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.add_artist(plt.Circle((0, 0), 1.0, color="black", fill=False))
    for i, w in enumerate(names):
        c0, c1, *rest = embeddings[i]
        x = c0
        y = c1
        ax.plot(x, y, "o", color="r")
        ax.text(x - 0.1, y + 0.04, re.sub("\.n\.\d{2}", "", w), color="b")
    fig.savefig("plots/" + name + ".png", dpi=fig.dpi)


Opts = namedtuple("Opts", "manifold dim negs batchsize ndproc burnin dampening")

if __name__ == "__main__":
    opt = Opts("poincare", 2, 50, 10, 4, 20, 0.75)
    manifold = MANIFOLDS[opt.manifold](debug=False, max_norm=500000)
    idx, objects, weights = load_edge_list("wordnet/mammal_closure.csv", False)
    # data = BatchedDataset(idx, objects, weights, opt.negs, opt.batchsize, opt.ndproc, opt.burnin > 0, opt.dampening)
    model, data, model_name, conf = initialize(
        manifold, opt, idx, objects, weights, sparse=False
    )
    model.load_weights("checkpoints/mammals-2d.tf")
    pplot(objects[:40], model.emb.numpy(), "tf-mammals-2d")
    plt.show()
