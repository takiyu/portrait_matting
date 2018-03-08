# -*- coding: utf-8 -*-
#
# Based on Chainer's `Evaluator`
#

import copy
import os
import warnings
import numpy as np
import six
import cv2
import multiprocessing

import chainer
from chainer import cuda
from chainer import configuration
from chainer import function
from chainer import link
from chainer import reporter as reporter_module
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.training import extension


def to_device(device, x):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def _save_vis(filename, mode, img, gt, score, alpha=None):
    # Score to labels
    label = np.argmax(score, axis=0)
    # Get original image format
    MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])
    img = img[0:3, :, :]  # Use only BGR
    img = img.transpose(1, 2, 0)  # C, H, W -> H, W, C
    img = (img + MEAN_BGR) / 255.0   # [-127:127] -> [0:1]

    def vis_segmentation(label_img):
        n_class = np.max(label_img) + 1
        vis_img = np.zeros(label_img.shape + (1, ), dtype=np.float32)
        for i in range(1, n_class):
            color = i / (n_class - 1)
            vis_img[label_img == i] = color
        return vis_img

    # Visualize
    h, w = gt.shape[0:2]
    fig = np.zeros((h * 2, w * 2, 3), dtype=np.float32)
    if mode.startswith('seg'):
        fig[0:h, 0:w, :] = vis_segmentation(gt)
        fig[0:h, w:, :] = vis_segmentation(label)
        fig[h:, 0:w] = img
    else:
        fig[0:h, 0:w, :] = gt.reshape(gt.shape + (1,))
        fig[0:h, w:, :] = alpha.reshape(alpha.shape + (1,))
        fig[h:, w:, :] = vis_segmentation(label)
        fig[h:, 0:w] = img

    # Save
    cv2.imwrite(filename, fig * 255)


def _vis_loop(inp_queue):
    while True:
        inp = inp_queue.get()
        if inp is None:
            break

        idx, mode, batch, preds, filename, iteration, epoch, base_dir = inp

        # Split batch
        if mode in ['seg', 'seg+', 'seg_tri']:
            imgs, gts = zip(*batch)
        elif mode == 'mat':
            imgs, gts, weight = zip(*batch)
        else:
            logger.error('Unknown mode ({})', mode)
            continue

        # Split predicitions
        if mode.startswith('seg'):
            scores = preds['score']
            alphas = [None] * len(scores)
        elif mode.startswith('mat'):
            scores = preds['score']
            alphas = preds['alpha']
        else:
            logger.error('Unknown mode ({})', mode)
            continue

        # Visualize
        for img, gt, score, alpha in zip(imgs, gts, scores, alphas):
            out_file = filename.format(index=idx, iteration=iteration,
                                       epoch=epoch)
            out_file = os.path.join(base_dir, out_file)
            # Save image
            _save_vis(out_file, mode, img, gt, score, alpha)
            idx += 1


class PortraitVisEvaluator(chainer.training.extension.Extension):

    """An extension that visualizes and evaluates output of a portrait model.

    Args:
        iterator: Iterator object that produces images and ground truth.
        target: Link object used for segmentation.
        label_names (iterable of str): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        filename (str): Basename for the saved image. It can contain two
            keywords, :obj:`'{iteration}'` and :obj:`'{index}'`. They are
            replaced with the iteration of the trainer and the index of
            the sample when this extension save an image. The default value is
            :obj:`'segmentation_iter={iteration}_idx={index}.jpg'`.
    """
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(
            self, iterator, target, device=None,
            converter=convert.concat_examples, label_names=None,
            filename='segmmentation_iter={iteration}_idx={index}.jpg',
            mode='seg', n_processes=None):

        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self.iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self.targets = target

        self.device = device
        self.converter = converter
        self.label_names = label_names
        self.filename = filename
        self.mode = mode
        self.n_processes = n_processes or multiprocessing.cpu_count()

    def __call__(self, trainer):
        # set up a reporter
        reporter = reporter_module.Reporter()
        if hasattr(self, 'name'):
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self.targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate(trainer)

        reporter_module.report(result)
        return result

    def evaluate(self, trainer):
        # Start workers
        inp_queue = multiprocessing.Queue()
        workers = list()
        for _ in range(self.n_processes):
            worker = multiprocessing.Process(target=_vis_loop,
                                             args=(inp_queue,), daemon=True)
            worker.start()
            workers.append(worker)

        iterator = self.iterators['main']
        eval_func = self.targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        idx = 0
        for batch in it:
            # Forward
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)
            summary.add(observation)

            # Get prediction
            if self.mode.startswith('seg'):
                preds = {'score': eval_func.score.data}
            elif self.mode.startswith('mat'):
                preds = {'score': eval_func.score.data,
                         'alpha': eval_func.alpha.data}
            else:
                logger.error('Invalid training mode')
                continue
            # Send to CPU
            for key in preds:
                preds[key] = to_device(-1, preds[key])

            # Visualize
            iteration = trainer.updater.iteration
            epoch = trainer.updater.epoch
            base_dir = trainer.out
            inp_queue.put((idx, self.mode, batch, preds, self.filename,
                          iteration, epoch, base_dir))

            idx += len(batch)

        # Exit
        for _ in range(self.n_processes):
            inp_queue.put(None)
        for worker in workers:
            worker.join()

        return summary.compute_mean()
