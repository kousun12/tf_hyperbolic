# Poincare Embeddings for TensorFlow

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
