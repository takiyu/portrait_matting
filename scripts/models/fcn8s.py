# -*- coding: utf-8 -*-
#
# Based on https://github.com/wkentaro/fcn
#  which is released under the MIT license.
#  https://opensource.org/licenses/mit-license.php
#

import os.path as osp

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from . import UpsamplingDeconvWeight


class FCN8s(chainer.Chain):

    def __init__(self, n_input_ch=3, n_output_ch=21, at_once=True):
        self.n_input_ch = n_input_ch
        self.n_output_ch = n_output_ch
        self.at_once = at_once
        kwargs = {
            'initialW': chainer.initializers.Zero(),
            'initial_bias': chainer.initializers.Zero(),
        }
        super().__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(n_input_ch, 64, 3, 1, 100, **kwargs)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, **kwargs)

            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1, **kwargs)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, **kwargs)

            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1, **kwargs)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, **kwargs)

            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1, **kwargs)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)

            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1, **kwargs)

            self.fc6 = L.Convolution2D(512, 4096, 7, 1, 0, **kwargs)
            self.fc7 = L.Convolution2D(4096, 4096, 1, 1, 0, **kwargs)

            self.score_fr = L.Convolution2D(4096, n_output_ch, 1, 1, 0,
                                            **kwargs)

            self.upscore2 = L.Deconvolution2D(
                n_output_ch, n_output_ch, 4, 2, 0, nobias=True,
                initialW=UpsamplingDeconvWeight())
            self.upscore8 = L.Deconvolution2D(
                n_output_ch, n_output_ch, 16, 8, 0, nobias=True,
                initialW=UpsamplingDeconvWeight())

            self.score_pool3 = L.Convolution2D(256, n_output_ch, 1, 1, 0,
                                               **kwargs)
            self.score_pool4 = L.Convolution2D(512, n_output_ch, 1, 1, 0,
                                               **kwargs)
            self.upscore_pool4 = L.Deconvolution2D(
                n_output_ch, n_output_ch, 4, 2, 0, nobias=True,
                initialW=UpsamplingDeconvWeight())

            # Disable up score's update
            self.upscore2.disable_update()
            self.upscore8.disable_update()
            self.upscore_pool4.disable_update()

    def __call__(self, x, t=None):
        score = self.forward(x)

        if t is None:
            assert not chainer.config.train
            return

        loss = F.softmax_cross_entropy(score, t, normalize=True)
        if np.isnan(float(loss.data)):
            raise ValueError('Loss is nan.')
        chainer.report({'loss': loss}, self)

        accuracy = F.accuracy(score, t)
        chainer.report({'accuracy': accuracy}, self)

        return loss

    def forward(self, x):
        ''' Forward FCN8s network and save to `self.score` '''

        # conv1
        h = F.relu(self.conv1_1(x))
        conv1_1 = h
        h = F.relu(self.conv1_2(conv1_1))
        conv1_2 = h
        h = F.max_pooling_2d(conv1_2, 2, stride=2, pad=0)
        pool1 = h  # 1/2

        # conv2
        h = F.relu(self.conv2_1(pool1))
        conv2_1 = h
        h = F.relu(self.conv2_2(conv2_1))
        conv2_2 = h
        h = F.max_pooling_2d(conv2_2, 2, stride=2, pad=0)
        pool2 = h  # 1/4

        # conv3
        h = F.relu(self.conv3_1(pool2))
        conv3_1 = h
        h = F.relu(self.conv3_2(conv3_1))
        conv3_2 = h
        h = F.relu(self.conv3_3(conv3_2))
        conv3_3 = h
        h = F.max_pooling_2d(conv3_3, 2, stride=2, pad=0)
        pool3 = h  # 1/8

        # conv4
        h = F.relu(self.conv4_1(pool3))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool4 = h  # 1/16

        # conv5
        h = F.relu(self.conv5_1(pool4))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2, pad=0)
        pool5 = h  # 1/32

        # fc6
        h = F.relu(self.fc6(pool5))
        h = F.dropout(h, ratio=.5)
        fc6 = h  # 1/32

        # fc7
        h = F.relu(self.fc7(fc6))
        h = F.dropout(h, ratio=.5)
        fc7 = h  # 1/32

        # score_fr
        h = self.score_fr(fc7)
        score_fr = h  # 1/32

        # score_pool3
        if self.at_once:
            pool3 *= 0.0001  # Scale to train at once
        h = self.score_pool3(pool3)
        score_pool3 = h  # 1/8

        # score_pool4
        if self.at_once:
            pool4 *= 0.01  # Scale to train at once
        h = self.score_pool4(pool4)
        score_pool4 = h  # 1/16

        # upscore2
        h = self.upscore2(score_fr)
        upscore2 = h  # 1/16

        # score_pool4c
        h = score_pool4[:, :,
                        5:5 + upscore2.shape[2],
                        5:5 + upscore2.shape[3]]
        score_pool4c = h  # 1/16

        # fuse_pool4
        h = upscore2 + score_pool4c
        fuse_pool4 = h  # 1/16

        # upscore_pool4
        h = self.upscore_pool4(fuse_pool4)
        upscore_pool4 = h  # 1/8

        # score_pool4c
        h = score_pool3[:, :,
                        9:9 + upscore_pool4.shape[2],
                        9:9 + upscore_pool4.shape[3]]
        score_pool3c = h  # 1/8

        # fuse_pool3
        h = upscore_pool4 + score_pool3c
        fuse_pool3 = h  # 1/8

        # upscore8
        h = self.upscore8(fuse_pool3)
        upscore8 = h  # 1/1

        # score
        h = upscore8[:, :, 31:31 + x.shape[2], 31:31 + x.shape[3]]
        score = h  # 1/1
        self.score = score

        return score

    def init_from_fcn8s(self, fcn8s):
        ''' Copy layer weights and biases from other FCN8s network '''

        l_names = ['conv1_2', 'conv1_1', 'conv2_1', 'conv2_2', 'conv3_1',
                   'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
                   'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7',
                   'score_pool4', 'score_pool3', 'score_fr', 'upscore2',
                   'upscore8', 'upscore_pool4']
        for l in self.children():
            # Ignore extra layers
            if l.name not in l_names:
                continue

            l1 = getattr(fcn8s, l.name)
            l2 = getattr(self, l.name)
            if l.name == 'conv1_1' and l1.W.shape != l2.W.shape:
                # Customized input layer (mean mask and grids)
                assert l1.W.shape[0] == l2.W.shape[0]
                assert l1.W.shape[2:] == l2.W.shape[2:]
                assert l1.W.shape[1] == 3
                assert l2.W.shape[1] == 6
                assert l1.b.shape == l2.b.shape
                l2.W.data[:, 0:3, :, :] = l1.W.data[...]
                l2.W.data[:, 3:, :, :] = \
                    np.mean(l1.W.data[...], axis=1, keepdims=True)
                l2.b.data[...] = l1.b.data[...]
            elif l.name.startswith('conv') or l.name.startswith('fc'):
                # conv or FC
                assert l1.W.shape == l2.W.shape
                assert l1.b.shape == l2.b.shape
                l2.W.data[...] = l1.W.data[...]
                l2.b.data[...] = l1.b.data[...]
            elif l.name.startswith('score'):
                # Score
                assert l1.W.shape[1:] == l2.W.shape[1:]
                if l1.W.shape == l2.W.shape:
                    # Same channels
                    assert l1.b.shape == l2.b.shape
                    l2.W.data[...] = l1.W.data[...]
                    l2.b.data[...] = l1.b.data[...]
                else:
                    # Different channels
                    if l1.W.data.shape[0] == 2 and self.n_output_ch == 3:
                        # PortraitFCN8s -> FCN8sMatting
                        l2.W.data[0, ...] = l1.W.data[0, ...]  # Back
                        l2.b.data[0] = l1.b.data[0]
                        l2.W.data[1, ...] = 0  # Unknown
                        l2.b.data[1] = 0
                        l2.W.data[2, ...] = l1.W.data[1, ...]  # Fore
                        l2.b.data[2] = l1.b.data[1]
                    else:
                        # Fill randomly
                        for dst_idx in range(self.n_output_ch):
                            src_idx = np.random.randint(l1.W.data.shape[0])
                            l2.W.data[dst_idx, ...] = l1.W.data[src_idx, ...]
                            l2.b.data[dst_idx] = l1.b.data[src_idx]
            elif l.name.startswith('upscore'):
                # Upscore layers are constant
                pass
            else:
                logger.error('Unknown layer name (%s)', l.name)

    def predict(self, imgs):
        with chainer.no_backprop_mode(), \
                chainer.using_config('train', False):
            self.forward(imgs)
            lbls = chainer.functions.argmax(self.score, axis=1)
        return lbls.data
