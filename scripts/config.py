# -*- coding: utf-8 -*-
import json

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def load(filename):
    ''' Load configure json file
    Loaded variables will be stored in config's globals
    '''
    logger.info('Load config from "{}"'.format(filename))
    with open(filename) as f:
        data = json.load(f)
        # parse json to python variables
        for key, value in data.items():
            if key in globals():
                logger.error('Conflict in config with key "{}"'.format(key))
            else:
                globals()[key] = value
