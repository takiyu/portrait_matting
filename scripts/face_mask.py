# -*- coding: utf-8 -*-

import cv2
import dlib
import os
import urllib.request
import numpy as np

# modules
from face_detector import FaceDetector

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def _load_imgs(dataset, n_data_use, face_detector):
    n_data = len(dataset)
    n_data_use = min(n_data_use, n_data)
    logger.info('Load random %d images from %d', n_data_use, n_data)

    data_list = list()
    for idx in np.random.permutation(n_data)[:n_data_use]:
        # Load
        img, mask = dataset[idx]
        # Detect faces
        lmk = face_detector(img)
        if lmk is None:
            logger.debug('Failed to detect any face')
            continue
        # Register
        data_list.append((img, mask, lmk))

    return data_list


def _generate_mean_landmark(data_list):
    logger.info('Generate mean landmark')

    # Accumulate landmarks
    mean_lmk = np.zeros_like(data_list[0][2])
    mean_lmk_denom = 0.0
    for _, _, lmk in data_list:
        mean_lmk += lmk
        mean_lmk_denom += 1.0

    # Average
    return mean_lmk / mean_lmk_denom


def _align_face_projection(img, src_lmk, dst_lmk, size=None):
    H, _ = cv2.findHomography(src_lmk, dst_lmk)
    if size is None:
        size = (img.shape[1], img.shape[0])
    # TODO: Border method
    aligned_img = cv2.warpPerspective(img, H, size,
                                      borderMode=cv2.BORDER_REPLICATE)
    return aligned_img


def _generate_mean_mask(data_list, mean_lmk):
    logger.info('Generate mean mask')

    # Accumulate masks
    mean_mask = np.zeros_like(data_list[0][1], dtype=np.float32)
    mean_mask_denom = 0.0
    for _, mask, lmk in data_list:
        # Align the projection using landmarks
        aligned_mask = _align_face_projection(mask, lmk, mean_lmk)
        # Add
        aligned_mask = aligned_mask.astype(np.float32) / 255.0
        mean_mask += aligned_mask
        mean_mask_denom += 1.0

    # Average
    mean_mask = mean_mask / mean_mask_denom
    mean_mask = (mean_mask * 255.0).astype(np.uint8)
    return mean_mask


def _generate_coord_img(size_x, size_y, grid_min, grid_max, pad_scale,
                        src_lmk, dst_lmk):
    pad_x = int(size_x * pad_scale)
    pad_y = int(size_y * pad_scale)
    pad_size_x = size_x + (pad_x * 2)
    pad_size_y = size_y + (pad_y * 2)

    grid_diff = (grid_max - grid_min) * pad_scale
    pad_grid_min = grid_min - grid_diff
    pad_grid_max = grid_max + grid_diff

    xs = np.linspace(pad_grid_min, pad_grid_max, pad_size_x)
    ys = np.linspace(pad_grid_min, pad_grid_max, pad_size_y)
    grid_x, grid_y = np.meshgrid(xs, ys)

    pad_src_lmk = src_lmk + np.array([pad_x, pad_y])
    size = (size_x, size_y)
    grid_x = _align_face_projection(grid_x, pad_src_lmk, dst_lmk, size)
    grid_y = _align_face_projection(grid_y, pad_src_lmk, dst_lmk, size)

    return grid_x, grid_y


class FaceMasker(object):

    def __init__(self, predictor_path, mask_filepath, dataset=None,
                 n_data_max=500):
        logger.info('Setup mean mask ("%s")', mask_filepath)
        self.mean_mask = None
        self.mean_lmk = None
        self.face_detector = FaceDetector(predictor_path)

        # Load cached mask
        if self._load_mean_mask(mask_filepath):
            return

        # Generate mean mask and landmark
        if dataset is None:
            logger.error('`dataset` is required for mask generation')
            return
        data_list = _load_imgs(dataset, n_data_max, self.face_detector)
        self.mean_lmk = _generate_mean_landmark(data_list)
        self.mean_mask = _generate_mean_mask(data_list, self.mean_lmk)

        # Save cache
        self._save_mean_mask(mask_filepath)

    def align(self, img, grid_min=-1.0, grid_max=1.0, pad_scale=5.0):
        # TODO: Tune grid scaling

        # Detect landmark
        lmk = self.face_detector(img)
        if lmk is None:
            return None

        # Mask
        mask = _align_face_projection(self.mean_mask, self.mean_lmk, lmk,
                                      (img.shape[1], img.shape[0]))

        # Grid image
        size_y, size_x = mask.shape
        grid_x, grid_y = _generate_coord_img(size_x, size_y, grid_min,
                                             grid_max, pad_scale,
                                             self.mean_lmk, lmk)

        return mask, grid_x, grid_y

    def _load_mean_mask(self, mask_filepath):
        if not os.path.exists(mask_filepath):
            return False
        data = np.load(mask_filepath)
        self.mean_mask = data['mean_mask']
        self.mean_lmk = data['mean_lmk']
        return True

    def _save_mean_mask(self, mask_filepath):
        np.savez(mask_filepath,
                 mean_mask=self.mean_mask,
                 mean_lmk=self.mean_lmk)
