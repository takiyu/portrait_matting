# -*- coding: utf-8 -*-
import glob
import os
import chainer
import cv2
import functools

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def split_dataset(dataset, test_per=0.05, seed=1234):
    # Split into train and test
    n_train = int(len(dataset) * (1.0 - test_per))
    train, test = chainer.datasets.split_dataset_random(dataset, n_train,
                                                        seed=seed)
    return train, test


def _get_name_set(dir_name, rm_ext=False):
    path_list = glob.glob(os.path.join(dir_name, '*'))
    name_set = set()
    for path in path_list:
        name = os.path.basename(path)
        if rm_ext:
            name = os.path.splitext(name)[0]
        name_set.add(name)
    return name_set


def _sort_names(names):
    names = list(names)
    names.sort()
    return names


def get_valid_names(*dirs, rm_exts=None):
    # Extract valid names
    if rm_exts is None:
        name_sets = [_get_name_set(d) for d in dirs]
    else:
        name_sets = [_get_name_set(d, r) for d, r in zip(dirs, rm_exts)]

    # Reduce
    def _join_and(a, b):
        return a & b

    valid_names = functools.reduce(_join_and, name_sets)
    if len(valid_names) == 0:
        logger.warn('no image is valid')
    else:
        logger.info('%d images are valid', len(valid_names))

    # Sort for the consistency
    return _sort_names(valid_names)
