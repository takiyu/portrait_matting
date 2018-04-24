#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import cv2
import numpy as np
import chainer

# modules
import log_initializer
import config
from .face_mask import FaceMasker
from . import models
from . import transforms

# logging
from logging import getLogger, INFO
log_initializer.set_fmt()
log_initializer.set_root_level(INFO)
logger = getLogger(__name__)


class Predictor(object):
    def __init__(self, mode, model_path, model_mode=None, device=None,
                 face_predictor_filepath=None, mean_mask_filepath=None):
        self._mode = mode
        self._device = device if device is not None else -1

        # Create empty model
        self._model = models.create(mode)

        if model_mode is None:
            # Load model
            chainer.serializers.load_npz(model_path, self._model)
        else:
            # Create temporary model and copy
            tmp_model = models.create(model_mode)
            chainer.serializers.load_npz(model_path, tmp_model)
            self._model.init_from_fcn8s(tmp_model)
            del tmp_model

        # Send to GPU
        if self._device >= 0:
            self._model.to_gpu(self._device)

        # Create transform function
        self._transform = transforms.create(mode)

        if mode in ['seg+', 'seg_tri', 'mat']:
            # Setup face masker
            self._face_masker = FaceMasker(face_predictor_filepath,
                                           mean_mask_filepath)

    def predict(self, img):
        # Create input array
        inp = [img]

        if self._mode in ['seg+', 'seg_tri', 'mat']:
            # Detect face
            ret_align = self._face_masker.align(img)
            if ret_align is None:
                logger.error('Failed to detect a face')
                return
            mean_mask, mean_grid_x, mean_grid_y = ret_align

            # Cast for saving storage
            mean_grid_x = mean_grid_x.astype(np.float32)
            mean_grid_y = mean_grid_y.astype(np.float32)

            # Append
            inp.extend([mean_mask, mean_grid_x, mean_grid_y, None])
            if self._mode == 'mat':
                inp.append(None)

        # Transform inputs
        inp = self._transform(inp)[0]  # Use only inputs (no teacher)
        inp = inp.reshape((1,) + inp.shape)
        inp = chainer.Variable(inp)

        # Send to GPU
        if self._device >= 0:
            inp.to_gpu(self._device)

        # Forward
        logger.info('Forward')
        with chainer.function.no_backprop_mode():
            out = self._model.forward(inp)

        # Send to CPU
        if self._device >= 0:
            out.to_cpu()

        # Extract batch
        out = out.data[0]
        return out  # Score or alpha


def main(argv):
    # Argument
    parser = argparse.ArgumentParser(description='Dataset Preparing Script')
    parser.add_argument('--config', '-c', default='config.json',
                        help='Load config from given json file')
    parser.add_argument('-i', required=True,
                        help='Input image file path')
    parser.add_argument('-o', default='output.png',
                        help='Output image file path')
    parser.add_argument('--mode', choices=['seg', 'seg+', 'seg_tri', 'mat'],
                        help='Model mode', required=True)
    parser.add_argument('--model_path', required=True,
                        help='Pretrained model path')
    parser.add_argument('--model_mode', default=None,
                        help='Mode for loading `model_path`')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args(argv)

    inp_filepath, out_filepath = args.i, args.o

    # Load config
    config.load(args.config)

    # Create predictor
    predictor = Predictor(args.mode, args.model_path, args.model_mode,
                          args.gpu, config.face_predictor_filepath,
                          config.mean_mask_filepath)

    # Load input image
    logger.info('Load input file: %s', inp_filepath)
    img = cv2.imread(inp_filepath)
    if img is None:
        logger.error('Failed to load')
        return

    # Predict
    ret = predictor.predict(img)
    if ret is None:
        logger.error('Failed to predict')
        return

    if args.mode.startswith('seg'):
        score = ret

        # Convert to trimap
        score = np.argmax(score, axis=0)

        # Write out trimap
        vis_img = np.zeros_like(img)
        vis_img[score == 1] = 127
        vis_img[score == 2] = 255
        cv2.imwrite(out_filepath, vis_img)

    elif args.mode.startswith('mat'):
        alpha = ret

        # Write out alpha
        cv2.imwrite(out_filepath, alpha * 255)


if __name__ == '__main__':
    main(sys.argv[1:])
