# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random


def _adjust_gamma(img, gamma=1.0):
    # Lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Apply
    return cv2.LUT(img, table)


def transform_random(inputs):
    # TODO: Make tunable
    rot_deg = random.gauss(0.0, 10.0)  # [-30:30]
    scale = abs(random.gauss(0.0, 0.1)) + 1.0  # [1.0: 1.3]
    gamma = random.uniform(0.7, 1.3)
    flip = random.choice([True, False])

    outputs = []

    h, w = inputs[0].shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, scale)
    for v in inputs:
        # Rotate and scale
        v = cv2.warpAffine(v, M, (w, h))
        # Flip
        if flip:
            v = cv2.flip(v, 1)
        outputs.append(v)

    # Gamma correction
    outputs[0] = _adjust_gamma(outputs[0], gamma)

    # Flip
    if flip and len(inputs) >= 5:
        # Invert mean_grid_x
        outputs[2] *= -1.0

    return outputs
