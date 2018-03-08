# -*- coding: utf-8 -*-
import numpy as np
import cv2

from . import _transform_img
from . import _transform_mask
from . import _transform_label
from . import _transform_label_tri
from . import _transform_alpha


def _create_seg_plus_input(img, mean_mask, mean_grid_x, mean_grid_y):
    # Input
    img = _transform_img(img)
    mean_mask = _transform_mask(mean_mask)
    # 6 ch
    inp = np.concatenate((img, [mean_mask], [mean_grid_x], [mean_grid_y]))
    return inp


def transform_seg(inputs):
    # Segmentation (BGR, Gray)
    assert len(inputs) == 2
    img, mask = inputs
    # Input
    img = _transform_img(img)
    # Output
    mask = _transform_label(mask)
    return img, mask


def transform_seg_plus(inputs):
    # Segmentation Plus (BGR, Gray, Gray, Gray, Gray)
    assert len(inputs) == 5
    # Input
    inp = _create_seg_plus_input(*inputs[:4])
    # Output
    mask = _transform_label(inputs[4])
    return inp, mask


def transform_seg_trimap(inputs):
    # Segmentation Plus (BGR, Gray, Gray, Gray, Gray)
    assert len(inputs) == 5
    # Input
    inp = _create_seg_plus_input(*inputs[:4])
    # Output
    mask = _transform_label_tri(inputs[4])
    return inp, mask


def transform_mat(inputs):
    # Matting (BGR, Gray, Gray, Gray, Gray, Gray)
    assert len(inputs) == 6
    # Input
    inp = _create_seg_plus_input(*inputs[:4])
    # Output
    alpha = _transform_alpha(inputs[4])
    weight = inputs[5]
    return inp, alpha, weight
