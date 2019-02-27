#!/usr/bin/env python3

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class RSGDTF(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        use_locking=False,
        name="RSGDTF",
        rgrad=None,
        expm=None,
    ):
        super(RSGDTF, self).__init__(use_locking, name)
        self.rgrad = rgrad
        self.expm = expm
        self._lr = learning_rate

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        d_p = self.rgrad(var, grad)
        return state_ops.assign_sub(var, self.expm(var, d_p, lr=lr_t))

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError("todo.")

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError("todo.")
