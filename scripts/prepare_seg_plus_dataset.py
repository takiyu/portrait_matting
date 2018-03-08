#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
#
# Prepare for Portrait FCN+.
#
#   Create mean mask and warp it.
#
# -----------------------------------------------------------------------------

import argparse
import os
import cv2
import numpy as np

# modules
import log_initializer
import config
from datasets import PortraitSegDataset, split_dataset, get_valid_names
from face_mask import FaceMasker

# logging
from logging import getLogger, INFO
log_initializer.set_fmt()
log_initializer.set_root_level(INFO)
logger = getLogger(__name__)


def align_mask(name, src_dir, dst_mask_dir, dst_grid_dir, face_masker):

    src_path = os.path.join(src_dir, name)
    if not os.path.exists(src_path):
        logger.error('"%s" dose not exist', src_path)
        return

    dst_mask_path = os.path.join(dst_mask_dir, name)
    dst_grid_path = os.path.join(dst_grid_dir, name + '.npz')
    if os.path.exists(dst_mask_path) and os.path.exists(dst_grid_path):
        logger.info('"%s" exists, Skip...', dst_mask_path)
        return

    logger.info('Align mean maks for "%s"', src_path)
    img = cv2.imread(src_path)
    ret_align = face_masker.align(img)
    if ret_align is None:
        logger.debug('Failed to detect a face')
        return
    mask, grid_x, grid_y = ret_align

    # Cast for saving storage
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    # Save
    cv2.imwrite(dst_mask_path, mask)
    np.savez(dst_grid_path, grid_x=grid_x, grid_y=grid_y)


def main():
    # Argument
    parser = argparse.ArgumentParser(description='Dataset Preparing Script')
    parser.add_argument('--config', '-c', default='config.json',
                        help='Load config from given json file')
    args = parser.parse_args()

    # Load config
    config.load(args.config)

    # Setup segmentation dataset
    dataset = PortraitSegDataset(config.img_crop_dir, config.img_mask_dir)
    # Split into train and test
    train_raw, _ = split_dataset(dataset)

    # Setup mean mask
    face_masker = FaceMasker(config.face_predictor_filepath,
                             config.mean_mask_filepath, train_raw)

    # Get valid names in 3 channel segmentation stage
    names = get_valid_names(config.img_crop_dir, config.img_mask_dir)

    # Start alignment
    logger.info('Generate aligned mask and grids')
    os.makedirs(config.img_mean_mask_dir, exist_ok=True)
    os.makedirs(config.img_mean_grid_dir, exist_ok=True)
    for name in names:
        align_mask(name, config.img_crop_dir, config.img_mean_mask_dir,
                   config.img_mean_grid_dir, face_masker)


if __name__ == '__main__':
    main()
