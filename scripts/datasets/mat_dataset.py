# -*- coding: utf-8 -*-
import glob
import os
import chainer
import cv2
import functools
import numpy as np
import scipy.sparse

# modules
from . import PortraitSegPlusDataset, get_valid_names

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class PortraitMattingDataset(PortraitSegPlusDataset):
    ''' Portrait Matting Dataset for FCN+ + closed form matting '''

    def __init__(self, crop_dir, mask_dir, mean_mask_dir, mean_grid_dir,
                 alpha_dir, alpha_weight_dir):
        self._crop_dir = crop_dir
        self._mask_dir = mask_dir
        self._mean_mask_dir = mean_mask_dir
        self._mean_grid_dir = mean_grid_dir
        self._alpha_dir = alpha_dir
        self._alpha_weight_dir = alpha_weight_dir
        self._names = get_valid_names(
            crop_dir, mask_dir, mean_mask_dir, mean_grid_dir, alpha_dir,
            alpha_weight_dir, rm_exts=[False, False, False, True, False, True])

    def get_example(self, index):
        name = self._names[index]

        # Load segmentation's data
        img, mean_mask, mean_grid_x, mean_grid_y, mask = \
            super().get_example(index)

        # Load alpha
        alpha_path = os.path.join(self._alpha_dir, name)
        alpha = cv2.imread(alpha_path, 0)  # BGR

        # Load alpha weight
        weight_path = os.path.join(self._alpha_weight_dir, name + '.npz')
        weight_data = np.load(weight_path)
        alpha_weight = weight_data['weight'].astype(np.float32)

        return img, mean_mask, mean_grid_x, mean_grid_y, alpha, alpha_weight
