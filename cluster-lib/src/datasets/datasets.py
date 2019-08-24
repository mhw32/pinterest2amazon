"""Loader for Images scrapped from Amazon."""

import os
import torch

from src.datasets.base import BaseDataset

AMAZON_DIR = '/mnt/fs5/wumike/datasets/pinterest2amazon/amazon'


def load_dataset(name, split='train', image_transforms=None):
    # Master function for loading data
    if name == 'amazon':
        return AmazonDataset(split=split, image_transforms=image_transforms)
    else:
        raise Exception('Dataset {} not recognized.'.format(name))


class AmazonDataset(BaseDataset):
    def __init__(self, split='train', image_transforms=None):
        super().__init__(
            AMAZON_DIR, split=split, image_transforms=image_transforms)


