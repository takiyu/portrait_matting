#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import chainer
from chainer import training
from chainer.dataset import convert
from chainer.training import extensions
import chainerui.extensions
import chainerui.utils
import sys

# modules
import log_initializer
import config
import custom_extensions
import custom_converters
import datasets
import models
import transforms

# logging
from logging import getLogger, INFO
log_initializer.set_fmt()
log_initializer.set_root_level(INFO)
logger = getLogger(__name__)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--config', '-c', default='config.json',
                        help='Configure json filepath')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--max_iteration', '-e', type=int, default=30000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpus', '-g', type=int, default=[-1], nargs='*',
                        help='GPU IDs (negative value indicates CPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--momentum', default=0.99, help='Momentum for SGD')
    parser.add_argument('--weight_decay', default=0.0005, help='Weight decay')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--mode', choices=['seg', 'seg+', 'seg_tri', 'mat'],
                        help='Training mode', required=True)
    parser.add_argument('--pretrained_fcn8s', default=None,
                        help='Pretrained model path of FCN8s')
    parser.add_argument('--pretrained_n_input_ch', default=3, type=int,
                        help='Input channel number of Pretrained model')
    parser.add_argument('--pretrained_n_output_ch', default=21, type=int,
                        help='Output channel number of Pretrained model')
    parser.add_argument('--mat_scale', default=4, type=int,
                        help='Matting scale for speed up')
    args = parser.parse_args(argv)
    return args


def setup_dataset(mode, crop_dir, mask_dir=None, mean_mask_dir=None,
                  mean_grid_dir=None, trimap_dir=None, alpha_dir=None,
                  alpha_weight_dir=None):
    # Create dataset
    dataset = datasets.create(mode, crop_dir, mask_dir, mean_mask_dir,
                              mean_grid_dir, trimap_dir, alpha_dir,
                              alpha_weight_dir)

    # Create transform function
    transform = transforms.create(mode)
    transform_random = transforms.transform_random

    # Split into train and test
    train_raw, test_raw = datasets.split_dataset(dataset)

    # Increase data variety
    train_raw = chainer.datasets.TransformDataset(train_raw, transform_random)

    # Transform for network inputs
    train = chainer.datasets.TransformDataset(train_raw, transform)
    test = chainer.datasets.TransformDataset(test_raw, transform)

    return train, test


def setup_iterators(gpus, batchsize, train_dataset, test_dataset):
    # Train
    if len(gpus) == 1:
        logger.info('Setup serial iterator')
        train_iter = chainer.iterators.MultiprocessIterator(train_dataset,
                                                            batchsize)
    else:
        logger.info('Setup multiprocess iterators')
        train_iter = \
            [chainer.iterators.MultiprocessIterator(i, batchsize)
             for i in chainer.datasets.split_dataset_n_random(train_dataset,
                                                              len(gpus))]
    # Test
    test_iter = chainer.iterators.MultiprocessIterator(test_dataset, batchsize,
                                                       repeat=False,
                                                       shuffle=False)

    return train_iter, test_iter


def setup_model(mode, pretrained_path=None, pretrained_n_input_ch=2,
                pretrained_n_output_ch=21, mat_scale=4):
    # Create empty model
    model = models.create(mode, mat_scale=mat_scale)

    # Copy from pretrained model
    if pretrained_path is not None:
        if mode == 'seg' or mode == 'seg+' or mode == 'seg_tri' or \
           mode == 'mat':
            # FCN8s
            logger.info('Load pretrained FCN8s model (%s)', pretrained_path)
            pretrained = models.FCN8s(n_input_ch=pretrained_n_input_ch,
                                      n_output_ch=pretrained_n_output_ch)
            chainer.serializers.load_npz(pretrained_path, pretrained)
            model.init_from_fcn8s(pretrained)

        else:
            logger.error('Unknown mode')

    return model


def setup_optimizer(model, lr, momentum, weight_decay):
    logger.info('Setup optimizer: lr=%f, momentum=%f, weight_decay=%f', lr,
                momentum, weight_decay)
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=momentum)
    optimizer.setup(model)

    # Learning rate hooks
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=weight_decay))
    for p in model.params():
        if p.name == 'b':
            p.update_rule = chainer.optimizers.momentum_sgd.MomentumSGDRule(
                lr=optimizer.lr * 2, momentum=0)

    return optimizer


def select_converter(mode='seg'):
    if mode in ['seg', 'seg+', 'seg_tri']:
        return convert.concat_examples  # Default
    elif mode == 'mat':
        return custom_converters.matting_converter
    else:
        logger.error('Unknown mode')


def setup_updater(mode, gpus, train_iter, optimizer):
    gpu0 = gpus[0]
    if len(gpus) == 1:
        # Single GPU or CPU
        logger.info('Setup single updater (gpu: %d)', gpu0)
        updater = training.StandardUpdater(train_iter, optimizer, device=gpu0,
                                           converter=select_converter(mode))
    else:
        # Multiple GPUs
        logger.info('Setup parallel updater (gpu: %s)', str(gpus))
        devs = {'slave{}'.format(i): gpu for i, gpu in enumerate(gpus[1:])}
        devs['main'] = gpu0
        updater = training.updaters.MultiprocessParallelUpdater(
            train_iter, optimizer, devices=devs,
            converter=select_converter(mode))
    return updater


def register_extensions(trainer, model, test_iter, args):
    if args.mode.startswith('seg'):
        # Max accuracy
        best_trigger = training.triggers.BestValueTrigger(
            'validation/main/accuracy', lambda a, b: a < b, (1, 'epoch'))
    elif args.mode.startswith('mat'):
        # Min loss
        best_trigger = training.triggers.BestValueTrigger(
            'validation/main/loss', lambda a, b: a > b, (1, 'epoch'))
    else:
        logger.error('Invalid training mode')

    # Segmentation extensions
    trainer.extend(
        custom_extensions.PortraitVisEvaluator(
            test_iter, model, device=args.gpus[0],
            converter=select_converter(args.mode),
            filename='vis_epoch={epoch}_idx={index}.jpg',
            mode=args.mode
        ), trigger=(1, 'epoch'))

    # Basic extensions
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport(trigger=(200, 'iteration')))
    trainer.extend(extensions.ProgressBar(update_interval=20))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy',
         'validation/main/accuracy', 'lr', 'elapsed_time']))
    trainer.extend(extensions.observe_lr(), trigger=(200, 'iteration'))

    # Snapshots
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}'
    ), trigger=(5, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, filename='model_best'
    ), trigger=best_trigger)

    # ChainerUI extensions
    trainer.extend(chainerui.extensions.CommandsExtension())
    chainerui.utils.save_args(args, args.out)

    # Plotting extensions
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'],
                'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))


def main(argv):
    # Argument
    args = parse_arguments(argv)

    # Load config
    config.load(args.config)

    # Setup dataset
    train, test = setup_dataset(args.mode, config.img_crop_dir,
                                config.img_mask_dir, config.img_mean_mask_dir,
                                config.img_mean_grid_dir,
                                config.img_trimap_dir, config.img_alpha_dir,
                                config.img_alpha_weight_dir)

    # Setup iterators
    train_iter, test_iter = setup_iterators(args.gpus, args.batchsize, train,
                                            test)

    # Setup model
    model = setup_model(args.mode, args.pretrained_fcn8s,
                        args.pretrained_n_input_ch,
                        args.pretrained_n_output_ch,
                        args.mat_scale)

    # Setup an optimizer
    optimizer = setup_optimizer(model, args.lr, args.momentum,
                                args.weight_decay)

    # Set up a trainer
    updater = setup_updater(args.mode, args.gpus, train_iter, optimizer)
    trainer = training.Trainer(updater, (args.max_iteration, 'iteration'),
                               out=args.out)

    # Register extensions for portrait segmentation / matting
    register_extensions(trainer, model, test_iter, args)

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main(sys.argv[1:])
