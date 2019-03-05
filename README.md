# Hyperbolic N-Space Embeddings for TensorFlow

Hyperbolic n-space, is a maximally symmetric, n-dimensional Riemannian manifold with constant negative sectional curvature. It turns out that hyperbolic space is very well suited for representing hierarchical data because it 'curves' space, allowing parent and sibling distances to stay constant over many branches without needing to increase dimensionality, as is the case in Euclidean space. A relatively intuitive model for thinking of embeddings in hyperbolic space is the poincare ball, where you can think of distances that increase exponentially as you move out from the center of the disk/ball.

This is an implementation of some basic functions for supporting hyperbolic geometries in the Poincare model (Lorentz to come) as well as functions to calculate riemann gradients over hyperbolic riemann manifolds. Loss is a simple cross-entropy as a demonstration.

<p align="center"><img src="mammals-2d.png" alt="plot"></p>
<p align="center"><img src="circle-limit.jpg" alt="escher"></p>

#### Getting Started

Build cython loader:

This will generate the transitive closure of the full noun hierarchy as well as of the mammals subtree of WordNet.

```bash
python setup.py build_ext --inplace
```

To embed the transitive closure of the WordNet mammals subtree, first generate the data via
```bash
cd wordnet
python transitive_closure.py
```

#### License
This code is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

![badge](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

In part adapted from [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)
