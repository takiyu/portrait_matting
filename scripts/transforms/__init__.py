# -*- coding: utf-8 -*-

from .common import _transform_img
from .common import _transform_mask
from .common import _transform_label
from .common import _transform_label_tri
from .common import _transform_alpha

from .random_transform import transform_random
from .dataset_transform import transform_seg
from .dataset_transform import transform_seg_plus
from .dataset_transform import transform_seg_trimap
from .dataset_transform import transform_mat

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def create(mode):
    # Create transform function
    if mode == 'seg':
        logger.info('Create transform function of segmentation')
        transform = transform_seg
    elif mode == 'seg+':
        logger.info('Create transform function of segmentation+')
        transform = transform_seg_plus
    elif mode == 'seg_tri':
        logger.info('Create transform function of trimap segmentation')
        transform = transform_seg_trimap
    elif mode == 'mat':
        logger.info('Create transform function of matting')
        transform = transform_mat
    else:
        logger.error('Invalid mode for transform function creation (%s)', mode)
        transform = None

    return transform
