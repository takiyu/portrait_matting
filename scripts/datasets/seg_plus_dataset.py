# -*- coding: utf-8 -*-
import glob
import os
import chainer
import cv2
import functools
import numpy as np

# modules
from . import PortraitSegDataset, get_valid_names

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class PortraitSegPlusDataset(PortraitSegDataset):
    ''' Portrait Segmentation Dataset for FCN+ '''

    def __init__(self, crop_dir, mask_dir, mean_mask_dir, mean_grid_dir):
        self._crop_dir = crop_dir
        self._mask_dir = mask_dir
        self._mean_mask_dir = mean_mask_dir
        self._mean_grid_dir = mean_grid_dir   # *.jpg.npz
        self._names = get_valid_names(crop_dir, mask_dir, mean_mask_dir,
                                      mean_grid_dir,
                                      rm_exts=[False, False, False, True])

    def get_example(self, index):
        name = self._names[index]

        # Load image and mask
        img, mask = super().get_example(index)

        # Load mean mask
        mean_mask_path = os.path.join(self._mean_mask_dir, name)
        mean_mask = cv2.imread(mean_mask_path, 0)  # Gray

        # Load grid image (fp32)
        mean_grid_path = os.path.join(self._mean_grid_dir, name + '.npz')
        grid_data = np.load(mean_grid_path)
        mean_grid_x = grid_data['grid_x'].astype(np.float32)
        mean_grid_y = grid_data['grid_y'].astype(np.float32)

        return img, mean_mask, mean_grid_x, mean_grid_y, mask
