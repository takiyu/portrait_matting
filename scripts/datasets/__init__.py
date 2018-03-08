# -*- coding: utf-8 -*-

from .common import split_dataset
from .common import get_valid_names

from .seg_dataset import PortraitSegDataset
from .seg_plus_dataset import PortraitSegPlusDataset
from .mat_dataset import PortraitMattingDataset

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def create(mode, crop_dir, mask_dir=None, mean_mask_dir=None,
           mean_grid_dir=None, trimap_dir=None, alpha_dir=None,
           alpha_weight_dir=None):
    # Create dataset wrapper
    if mode == 'seg':
        logger.info('Create segmentation dataset')
        dataset = PortraitSegDataset(crop_dir, mask_dir)
    elif mode == 'seg+':
        logger.info('Create segmentation+ dataset')
        dataset = PortraitSegPlusDataset(crop_dir, mask_dir, mean_mask_dir,
                                         mean_grid_dir)
    elif mode == 'seg_tri':
        logger.info('Create segmentation+ dataset for trimap')
        dataset = PortraitSegPlusDataset(crop_dir, trimap_dir, mean_mask_dir,
                                         mean_grid_dir)
    elif mode == 'mat':
        logger.info('Create matting dataset')
        dataset = PortraitMattingDataset(crop_dir, mask_dir, mean_mask_dir,
                                         mean_grid_dir, alpha_dir,
                                         alpha_weight_dir)
    else:
        logger.error('Invalid mode for dataset creation (%s)', mode)
        dataset = None

    return dataset
