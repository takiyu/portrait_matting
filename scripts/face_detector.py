# -*- coding: utf-8 -*-

import bz2
import dlib
import os
import urllib.request
import numpy as np

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class FaceDetector(object):

    ''' Facial landmark detector currently using Dlib. '''

    predictor_url = \
        'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

    def __init__(self, predictor_path):
        logger.info('Setup face detector ("%s")', predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = self._setup_predictor(predictor_path)

    def __call__(self, img):
        ''' Detect facial landmarks of the largest face. '''
        # Detect the largest face rectangle
        rects = self.detector(img, 1)
        if len(rects) == 0:
            return None
        max_area, rect_idx = 0.0, -1
        for i, rect in enumerate(rects):
            area = rect.area()
            if max_area < area:
                max_area = area
                rect_idx = i

        # Detect landmark
        shape = self.predictor(img, rects[rect_idx])
        landmark = np.empty((len(shape.parts()), 2), np.float32)
        for i, p in enumerate(shape.parts()):
            landmark[i, 0], landmark[i, 1] = p.x, p.y

        return landmark

    def _setup_predictor(self, predictor_path):
        if not os.path.exists(predictor_path):
            # Download predictor file
            url = FaceDetector.predictor_url
            bz2_path = os.path.basename(url)

            logger.info('Download to "%s"', bz2_path)
            urllib.request.urlretrieve(url, bz2_path)

            logger.info('Expand to "%s"', predictor_path)
            data = bz2.BZ2File(bz2_path).read()
            open(predictor_path, 'wb').write(data)

            logger.info('Remove temporary file ("%s")', bz2_path)
            os.remove(bz2_path)

        # Deserialize
        predictor = dlib.shape_predictor(predictor_path)

        return predictor
