# -*- coding: utf-8 -*-

from .upsample_initializer import UpsamplingDeconvWeight
from .matting_link import MattingLink
from .laplacian import matting_laplacian

from .fcn8s import FCN8s
from .fcn8s_matting import FCN8sMatting

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def create(mode, mat_scale=1):
    # Create empty model
    if mode == 'seg':
        logger.info('Create FCN8s for segmentation')
        model = FCN8s(n_input_ch=3, n_output_ch=2)
    elif mode == 'seg+':
        logger.info('Create FCN8s for segmentation+')
        model = FCN8s(n_input_ch=6, n_output_ch=2)
    elif mode == 'seg_tri':
        logger.info('Create FCN8s for trimap segmentation')
        model = FCN8s(n_input_ch=6, n_output_ch=3)
    elif mode == 'mat':
        logger.info('Create FCN8s-Matting')
        model = FCN8sMatting(n_input_ch=6, mat_scale=mat_scale)
    else:
        logger.error('Invalid mode for model creation (%s)', mode)
        model = None

    return model
