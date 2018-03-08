# -*- coding: utf-8 -*-
#
# Based on https://github.com/wkentaro/fcn
#  which is released under the MIT license.
#  https://opensource.org/licenses/mit-license.php
#

import chainer
import numpy as np


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def _get_upsampling_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    return filt


class UpsamplingDeconvWeight(chainer.initializer.Initializer):

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        xp = chainer.cuda.get_array_module(array)

        in_c, out_c, kh, kw = array.shape
        assert in_c == out_c
        assert kh == kw

        filt = _get_upsampling_filter(kh)
        filt = xp.asarray(filt)

        array[...] = 0
        array[range(in_c), range(in_c), :, :] = filt
