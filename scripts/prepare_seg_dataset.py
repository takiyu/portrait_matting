#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
#
# Prepare for Portrait FCN.
#
#   Paper's dataset will be downloaded and converted for training.
#
# -----------------------------------------------------------------------------

import argparse
import os
import cv2
import glob
import urllib.request
import PIL.Image
import scipy.io

# modules
import log_initializer
import config

# logging
from logging import getLogger, INFO
log_initializer.set_fmt()
log_initializer.set_root_level(INFO)
logger = getLogger(__name__)


def load_img_urls(filepath):
    # Load file
    logger.info('Load image urls from "%s"', filepath)
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')

    # Parse for each line
    url_pairs = list()
    for line in lines:
        items = line.split()
        if len(items) != 2:
            logger.error('Invalid line. (%s)', line)
            continue
        if items[1] != 'None':
            # Register
            url_pairs.append(items)

    return url_pairs


def download_img(url, img_name, base_dir):
    img_path = os.path.join(base_dir, img_name)
    if os.path.exists(img_path):
        logger.info('"%s" exists. Skip...', img_path)
        return

    logger.info('Download to "%s"', img_path)
    try:
        urllib.request.urlretrieve(url, img_path)
    except urllib.error.HTTPError:
        logger.warin('Failed to download')


def load_crop_rects(filepath):
    # Load file
    logger.info('Load crop rectangles from "%s"', filepath)
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')

    # Parse for each line
    rect_pairs = list()
    for line in lines:
        items = line.split()
        if len(items) != 5:
            logger.error('Invalid line. (%s)', line)
            continue
        # Register
        rect_pairs.append((items[0], items[1:5]))

    return rect_pairs


def crop_img(img_name, src_dir, dst_dir, crop_rect, img_size):
    src_path = os.path.join(src_dir, img_name)
    if not os.path.exists(src_path):
        logger.error('"%s" dose not exist', src_path)
        return

    dst_path = os.path.join(dst_dir, img_name)
    if os.path.exists(dst_path):
        logger.info('"%s" exists, Skip...', dst_path)
        return

    logger.info('Crop "%s" to "%s"', src_path, dst_path)
    img = cv2.imread(src_path)
    x0, y0 = int(crop_rect[2]), int(crop_rect[0])
    x1, y1 = int(crop_rect[3]), int(crop_rect[1])
    img = img[y0:y1, x0:x1, :]
    img = cv2.resize(img, img_size)
    cv2.imwrite(dst_path, img)


def parse_mask(mask_name, src_dir, img_name, dst_dir):
    src_path = os.path.join(src_dir, mask_name)
    if not os.path.exists(src_path):
        logger.error('"%s" dose not exist', src_path)
        return

    dst_path = os.path.join(dst_dir, img_name)
    if os.path.exists(dst_path):
        logger.info('"%s" exists, Skip...', dst_path)
        return

    logger.info('Parse mask "%s" to "%s"', src_path, dst_path)
    img = scipy.io.loadmat(src_path)['mask']
    img *= 255  # [0:1] -> [0:255]
    cv2.imwrite(dst_path, img)


def main():
    # Argument
    parser = argparse.ArgumentParser(description='Dataset Preparing Script')
    parser.add_argument('--config', '-c', default='config.json',
                        help='Load config from given json file')
    args = parser.parse_args()

    # Load config
    config.load(args.config)

    # Load image urls
    url_pairs = load_img_urls(config.org_imgurl_filepath)
    # Download
    os.makedirs(config.img_raw_dir, exist_ok=True)
    for name, url in url_pairs:
        download_img(url, name, config.img_raw_dir)

    # Load crop rectangles
    rect_pairs = load_crop_rects(config.org_crop_filepath)
    # Crop
    img_size = (600, 800)  # Decided by mask size
    os.makedirs(config.img_crop_dir, exist_ok=True)
    for name, rect in rect_pairs:
        crop_img(name, config.img_raw_dir, config.img_crop_dir, rect, img_size)

    # Parse masks
    os.makedirs(config.img_mask_dir, exist_ok=True)
    for name, _ in rect_pairs:
        mask_name = '{}_mask.mat'.format(os.path.splitext(name)[0])
        parse_mask(mask_name, config.org_mask_dir, name, config.img_mask_dir)


if __name__ == '__main__':
    main()
