# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom datasets."""
import os
from typing import List, Callable, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
import rasterio


def _list_dict_to_dict_list(samples):
    """Convert a list of dictionaries to a dictionary of lists.

    Args:
        samples: a list of dictionaries

    Returns:
        a dictionary of lists
    """
    collated = dict()
    for sample in samples:
        for key, value in sample.items():
            if key not in collated:
                collated[key] = []
            collated[key].append(value)
    return collated


def stack_samples(samples):
    """Stack a list of samples along a new axis.

    Useful for forming a mini-batch of samples to pass to
    :class:`torch.utils.data.DataLoader`.

    Args:
        samples: list of samples

    Returns:
        a single sample
    """
    collated = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], torch.Tensor):
            collated[key] = torch.stack(value)
    return collated


class TileDataset(Dataset):
    def __init__(self, image_fns: List[str], mask_fns: List[str], transforms=None, sanity_check=True):
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        if self.mask_fns is not None:
            assert len(image_fns) == len(mask_fns)

        # Check to make sure that all the image and mask tile pairs are the same size
        # as a sanity check
        if sanity_check and mask_fns is not None:
            print("Running sanity check on dataset...")
            for image_fn, mask_fn in list(zip(image_fns, mask_fns)):
                with rasterio.open(image_fn[0]) as f:
                    image_height, image_width = f.shape
                with rasterio.open(mask_fn) as f:
                    mask_height, mask_width = f.shape
                assert image_height == mask_height
                assert image_width == mask_width

        self.transforms = transforms

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        i, y, x, patch_size = index

        sample = {
            "y": y,
            "x": x,
        }

        window = rasterio.windows.Window(x, y, patch_size, patch_size)

        # Load imagery
        stack = []
        for j in range(len(self.image_fns[i])):
            image_fn = self.image_fns[i][j]
            with rasterio.open(image_fn) as f:
                image = f.read(window=window)
            stack.append(image)
        stack = np.concatenate(stack, axis=0)
        sample["image"] = torch.from_numpy(stack).float()

        # Load mask
        if self.mask_fns is not None:
            mask_fn = self.mask_fns[i]
            with rasterio.open(mask_fn) as f:
                mask = f.read(window=window)
            sample["mask"] = torch.from_numpy(mask).long()

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample