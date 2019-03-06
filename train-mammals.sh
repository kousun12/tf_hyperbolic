#!/bin/sh

python3 embed.py \
       -dim 5 \
       -lr 0.3 \
       -epochs 300 \
       -negs 50 \
       -burnin 20 \
       -manifold poincare \
       -dset wordnet/mammal_closure.csv \
       -checkpoint checkpoints/mammals-5d.tf \
       -batchsize 20 \
       -eval_each 1
