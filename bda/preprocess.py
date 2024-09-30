# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Class and methods to use for preprocessing imagery in torchgeo datasets."""

from typing import Any

import torch
from torchvision.transforms import Normalize


class Preprocessor(object):
    """Class that runs basic preprocessing in training or inference mode."""

    def __init__(self, training_mode: bool, means: list[float], stds: list[float]):
        """Initialize the Preprocessor class.

        Args:
            training_mode (bool): Whether to run in training mode or not.
            means (list[float]): List of means to use for normalization.
            stds (list[float]): List of standard deviations to use for normalization.
        """
        self.training_mode = training_mode
        self.normalize = Normalize(means, stds)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Run preprocessing on a sample.

        Args:
            sample (dict[str, Any]): The sample to preprocess.

        Returns:
            dict[str, Any]: The preprocessed sample.
        """
        if "image" in sample:
            sample["image"] = torch.clip(self.normalize(sample["image"]), 0, 1)
        if "mask" in sample:
            sample["mask"] = sample["mask"].squeeze().long()

        # We remove the bounding box when training as it causes problems in lightning
        # based trainers
        if self.training_mode and "bbox" in sample:  # for torchgeo < 0.6
            del sample["bbox"]
        if self.training_mode and "bounds" in sample:  # for torchgeo >= 0.6
            del sample["bounds"]
        return sample
