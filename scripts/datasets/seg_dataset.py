# -*- coding: utf-8 -*-
import glob
import os
import chainer
import cv2
import functools
import numpy as np

# modules
from . import get_valid_names

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class PortraitSegDataset(chainer.dataset.DatasetMixin):
    ''' Portrait Segmentation Dataset for FCN'''

    def __init__(self, crop_dir, mask_dir):
        self._crop_dir = crop_dir
        self._mask_dir = mask_dir
        self._names = get_valid_names(crop_dir, mask_dir)

    def __len__(self):
        return len(self._names)

    def get_example(self, index):
        name = self._names[index]

        # Load image
        crop_path = os.path.join(self._crop_dir, name)
        img = cv2.imread(crop_path, 1)  # BGR

        # Load mask
        mask_path = os.path.join(self._mask_dir, name)
        mask = cv2.imread(mask_path, 0)  # Gray

        return img, mask
