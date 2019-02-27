#!/bin/sh

python3 embed.py \
       -dim 2 \
       -lr 0.3 \
       -epochs 300 \
       -negs 50 \
       -burnin 20 \
       -manifold poincare \
       -dset wordnet/mammal_closure.csv \
       -checkpoint checkpoints/mammals-2d.tf \
       -batchsize 10 \
       -eval_each 1 \
       -train_threads 1
