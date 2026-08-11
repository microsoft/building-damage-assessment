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
    def __init__(
        self,
        image_fns: List[str],
        mask_fns: List[str],
        transforms=None,
        sanity_check=True,
        num_channels: int = None,
        preload: bool = False,
    ):
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        self.num_channels = num_channels
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

        # Reading a patch from a compressed GeoTIFF forces GDAL to decompress every
        # block the patch overlaps, which dominates the cost of a training step. The
        # tiles are small enough to hold in memory, so we read them once up front and
        # crop from the resulting arrays instead.
        self.image_cache = None
        self.mask_cache = None
        if preload:
            self._preload()

    def _preload(self):
        """Read every tile into memory so patches can be cropped without decoding."""
        print("Preloading tiles into memory...")
        self.image_cache = []
        for fns in self.image_fns:
            stack = []
            for fn in fns:
                with rasterio.open(fn) as f:
                    stack.append(f.read())
            self.image_cache.append(np.concatenate(stack, axis=0))

        if self.mask_fns is not None:
            self.mask_cache = []
            for fn in self.mask_fns:
                with rasterio.open(fn) as f:
                    self.mask_cache.append(f.read())

        num_bytes = sum(a.nbytes for a in self.image_cache)
        if self.mask_cache is not None:
            num_bytes += sum(a.nbytes for a in self.mask_cache)
        print(f"Preloaded {len(self.image_fns)} tiles ({num_bytes / 1e6:.0f} MB)")

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        i, y, x, patch_size = index

        sample = {
            "y": y,
            "x": x,
        }

        # Load imagery
        if self.image_cache is not None:
            stack = self.image_cache[i][:, y : y + patch_size, x : x + patch_size].copy()
        else:
            window = rasterio.windows.Window(x, y, patch_size, patch_size)
            stack = []
            for j in range(len(self.image_fns[i])):
                image_fn = self.image_fns[i][j]
                with rasterio.open(image_fn) as f:
                    image = f.read(window=window)
                stack.append(image)
            stack = np.concatenate(stack, axis=0)
        if self.num_channels is not None:
            stack = stack[:self.num_channels]
        sample["image"] = torch.from_numpy(stack).float()

        # Load mask
        if self.mask_fns is not None:
            if self.mask_cache is not None:
                mask = self.mask_cache[i][
                    :, y : y + patch_size, x : x + patch_size
                ].copy()
            else:
                mask_fn = self.mask_fns[i]
                with rasterio.open(mask_fn) as f:
                    mask = f.read(window=rasterio.windows.Window(x, y, patch_size, patch_size))
            sample["mask"] = torch.from_numpy(mask).long()

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
