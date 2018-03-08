# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def _transform_img(img):
    # LSVRC2012 used by VGG16
    MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])

    img = img.astype(np.float32)  # BGR
    img -= MEAN_BGR
    img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W
    return img


def _transform_mask(mask):
    mask = mask.astype(np.float32)  # Gray
    mask -= 127.0
    return mask


def _transform_label(mask):
    if mask is None:
        return None
    lbl = np.zeros(mask.shape, np.int32)  # Gray
    lbl[127 <= mask] = 1
    return lbl


def _transform_label_tri(mask):
    if mask is None:
        return None
    lbl = np.zeros(mask.shape, np.int32)  # Gray
    lbl[85 <= mask] = 1
    lbl[170 <= mask] = 2
    return lbl


def _transform_alpha(alpha):
    if alpha is None:
        return None
    alpha = alpha.astype(np.float32)
    return alpha / 255
