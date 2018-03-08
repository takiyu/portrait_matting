# -*- coding: utf-8 -*-
#
# Based on https://github.com/wkentaro/fcn
#

import os.path as osp

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

# modules
from . import matting_laplacian
from . import FCN8s
from . import MattingLink

from logging import getLogger
logger = getLogger(__name__)


def _compute_laplacians(xs, prob_bs, prob_fs):
    # LSVRC2012 used by VGG16
    MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])

    laplacians = list()
    for x, prob_b, prob_f in zip(xs.data, prob_bs.data, prob_fs.data):
        # Get original image format
        x = x[0:3, :, :]  # Use only BGR
        img = x.transpose(1, 2, 0)  # C, H, W -> H, W, C
        img = (img + MEAN_BGR) / 255.0   # [-127:127] -> [0:1]

        # Constant map  (Remove extra channel of (1, h, w))
        consts_map = (0.9 < prob_b[0]) | (0.9 < prob_f[0])

        laplacian = matting_laplacian(img, ~consts_map)
        laplacians.append(laplacian)
    return np.asarray(laplacians)


class FCN8sMatting(FCN8s):

    def __init__(self, n_input_ch=6, at_once=True, mat_scale=1):
        self.mat_scale = mat_scale
        super().__init__(n_input_ch, 3, at_once)  # 3 outputs
        with self.init_scope():
            self.prob_scale = L.Scale(W_shape=(1,))
            self.prob_scale.W.copydata(np.array([100.0]))  # Initial parameter
            self.matting_link = MattingLink()

    def to_gpu(self, device=None):
        ''' Send layers to GPU except matting_link. '''
        with chainer.cuda._get_device(device):
            super(chainer.Chain, self).to_gpu()
            d = self.__dict__
            for name in self._children:
                if name == 'matting_link':
                    logger.info('Skip sending matting_link to GPU')
                    continue
                d[name].to_gpu()
        return self

    def __call__(self, x, t=None, w=None):
        # t, w is on host.

        # Forward network
        alpha = self.forward(x)

        if t is None:
            assert not chainer.config.train
            return

        # Weighted mean squared error
        # TODO: Do more tests
#         loss = F.mean(F.squared_error(alpha, t) * w)
        loss = F.mean_squared_error(alpha, t)

        if np.isnan(float(loss.data)):
            raise ValueError('Loss is nan.')
        chainer.report({'loss': loss}, self)

        return loss

    def forward(self, x):
        # FIXME: Adding with constant and F.resize_images() using 0th device

        # Forward FCN
        score = super().forward(x)

        # Convert score to probability
        prob = F.softmax(score, axis=1)

        # Increase gradient of the probability
        prob = prob - 0.5
        prob = self.prob_scale(prob)
        prob = F.clip(prob, -0.5, 0.5)
        prob = prob + 0.5

        # Down sampling
        h, w = x.shape[2:4]
        down_shape = (h // self.mat_scale, w // self.mat_scale)
        prob = F.resize_images(prob, down_shape)
        x = F.resize_images(x, down_shape)

        # Split into foreground, background and unknown sores
        prob_b, _, prob_f = F.split_axis(prob, 3, axis=1)   # (n, 1, h, w)

        # Copy to CPU
        x = F.copy(x, -1)
        prob_b = F.copy(prob_b, -1)
        prob_f = F.copy(prob_f, -1)

        # Compute laplacian
        laplacian = _compute_laplacians(x, prob_b, prob_f)

        # Matting
        alpha = self.matting_link(prob_b, prob_f, laplacian)  # (n, 1, h, w)

        # Up sampling
        alpha = F.resize_images(alpha, (h, w))

        # Remove extra channel (n, 1, h, w) -> (n, h, w)
        alpha_shape = (alpha.shape[0], alpha.shape[2], alpha.shape[3])
        alpha = F.reshape(alpha, alpha_shape)

        self.alpha = alpha
        return alpha

    def predict(self, imgs):
        with chainer.no_backprop_mode(), \
                chainer.using_config('train', False):
            self.forward(imgs)
        return self.alpha.data
