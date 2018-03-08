import functools
import operator
import numpy as np
from scipy import sparse

import chainer
from chainer import cuda
from chainer import function_node
from chainer import link
from chainer import variable
from chainer.utils import type_check

from logging import getLogger
logger = getLogger(__name__)


def _diag(x):
    return sparse.diags(x.reshape(-1))


def _solve(A, b):

    def _solve_drt(A, b):
        # Direct solver with umfpack. (float64 is required)
        solution = sparse.linalg.spsolve(A.astype(np.float64),
                                         b.astype(np.float64))
        return solution

    def _solve_itr(A, b):
        # Iterative solver (TODO: Tune maxiter)
        solution = sparse.linalg.bicg(A, b, maxiter=len(b))
        return solution[0]

    # Use iterative solver
#     return _solve_itr(A, b)
#     # Use direct solver
    return _solve_drt(A, b)


class MattingFunction(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 4)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[0].shape == in_types[1].shape
        )
        type_check.expect(
            isinstance(in_types[2][0], sparse.coo_matrix),
            in_types[0].shape[0] == len(in_types[2])
        )
        type_check.expect(in_types[3].shape == (1,))

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1, 2, 3))
        self.retain_outputs((0,))

        prob_b, prob_f, laplacians, lambda_ = inputs

        img_shape = prob_b[0].shape
        nm = prob_b[0].size
        lambda_f = float(lambda_)

        ret = []
        for b, f, laplacian in zip(prob_b, prob_f, laplacians):
            BF_diag = _diag(b + f)
            F = f.reshape(-1)

            D = sparse.csc_matrix(BF_diag * lambda_f + laplacian)
            alpha = _solve(D, F * lambda_f)

            # Reshape to the original
            ret.append(alpha.reshape(img_shape))

        return np.asarray(ret, dtype=np.float32),

    def forward_gpu(self, inputs):
        raise NotImplementedError

    def backward(self, indexes, gy):
        # TODO: Tune about sparse matrix structure

        # alpha = lambda_ * D^-1 * f
        prob_b, prob_f, laplacians, lambda_ = self.get_retained_inputs()
        alpha, = self.get_retained_outputs()

        # Laplacian is not np.array, so Variable wrapper cannot be used
        laps = laplacians.data
        prob_b = prob_b.data
        prob_f = prob_f.data
        lambda_ = lambda_.data
        alphas = alpha.data
        gy0s = gy[0].data

        img_shape = prob_b[0].shape
        nm = prob_b[0].size
        lambda_f = float(lambda_)

        ret0, ret1, ret2, ret3 = [], [], [], []
        for b, f, lap, alpha, gy0 in zip(prob_b, prob_f, laps, alphas, gy0s):
            BF_diag = _diag(b + f)
            F = f.reshape(-1)
            gY = gy0.reshape(-1)
            alpha = alpha.reshape(-1)

            D = sparse.csc_matrix(BF_diag * lambda_f + lap)
            D_inv_F_lambda = alpha

            if 0 in indexes or 1 in indexes:
                gb = _solve(D, -lambda_f * (_diag(D_inv_F_lambda) * gY))
                if 0 in indexes:
                    ret0.append(gb.reshape(img_shape))
                if 1 in indexes:
                    gf = gb + _solve(D, lambda_f * gY)
                    ret1.append(gf.reshape(img_shape))
            if 2 in indexes:
                # Gradient of laplacian is not needed.
                raise NotImplementedError
            if 3 in indexes:
                gl = _solve(D, -BF_diag * D_inv_F_lambda)
                gl += D_inv_F_lambda / lambda_f
                gl = gl.dot(gY)
                ret3.append(gl)

        ret = []
        if len(ret0) > 0:
            ret.append(chainer.Variable(np.asarray(ret0, dtype=np.float32)))
        if len(ret1) > 0:
            ret.append(chainer.Variable(np.asarray(ret1, dtype=np.float32)))
        if len(ret2) > 0:
            ret.append(chainer.Variable(np.asarray(ret2, dtype=np.float32)))
        if len(ret3) > 0:
            ret3 = np.sum(ret3).reshape(1)
            ret.append(chainer.Variable(np.asarray(ret3, dtype=np.float32)))
        return ret


def alpha_matting(prob_b, prob_f, laplacian, lambda_):
    return MattingFunction().apply((prob_b, prob_f, laplacian, lambda_))[0]


class MattingLink(link.Link):
    """Matting layer.

    Args:
        prob_b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable for background probability
        prob_f (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Input variable for foreground probability
        laplacian (:class:`~scipy.sparse.coo_matrix`): Sparse laplacian matrix

    Returns:
        ~chainer.Variable: Output variable. Alpha.

    """

    def __init__(self, init_lambda=100):
        super().__init__()
        with self.init_scope():
            self.lambda_ = variable.Parameter(init_lambda, shape=(1,))

    def _initialize_params(self, in_size):
        self.lambda_.initialize((1,))

    def __call__(self, prob_b, prob_f, laplacian):
        return alpha_matting(prob_b, prob_f, laplacian, self.lambda_)
