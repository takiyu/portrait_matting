import six
from chainer.dataset import convert
import numpy as np


def matting_converter(batch, device=None, padding=None):
    first_elem = batch[0]
    assert isinstance(first_elem, tuple)
    assert len(first_elem) == 3

    # Collect pure GPU variables (img)
    gpu_batch = [v[0] for v in batch]
    gpu_batch = convert.concat_examples(gpu_batch, device, padding)

    # Collect pure CPU variables (alpha, weight)
    cpu_batch = [v[1:] for v in batch]
    cpu_batch = convert.concat_examples(cpu_batch, -1, padding)

    # Concatenate
    return (gpu_batch, *cpu_batch)
