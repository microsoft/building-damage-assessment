# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom torchgeo datasets."""

import os
from typing import Callable, Optional

from torchgeo.datasets import RasterDataset


class SingleRasterDataset(RasterDataset):
    """A torchgeo dataset that loads a single raster file."""

    def __init__(self, fn: str, transforms: Optional[Callable] = None):
        """Initialize the SingleRasterDataset class.

        Args:
            fn (str): The path to the raster file.
            transforms (Optional[Callable], optional): The transforms to apply to the
                raster file. Defaults to None.
        """
        self.filename_regex = os.path.basename(fn)
        super().__init__(paths=os.path.dirname(fn), transforms=transforms)
