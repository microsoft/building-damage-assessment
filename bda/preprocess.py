# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Class and methods to use for preprocessing imagery in torchgeo datasets."""

from typing import Any, Optional, Sequence

import torch
from torchvision.transforms import Normalize


class Preprocessor(object):
    """Class that runs basic preprocessing in training or inference mode."""

    def __init__(
        self,
        training_mode: bool,
        means: list[float],
        stds: list[float],
        clip_range: Optional[Sequence[float]] = (0, 1),
    ):
        """Initialize the Preprocessor class.

        Args:
            training_mode (bool): Whether to run in training mode or not.
            means (list[float]): List of means to use for normalization.
            stds (list[float]): List of standard deviations to use for normalization.
            clip_range (Optional[Sequence[float]]): ``(low, high)`` bounds applied to
                the normalized imagery, or ``None`` to skip clipping. The default of
                ``(0, 1)`` suits min/max style values (e.g. means of 0 and stds of
                the data range). It is *destructive* for true mean/std
                standardization, where it would flatten every below-mean pixel to
                ``low``; use ``(-3, 3)`` or ``None`` in that case.

        Raises:
            ValueError: If *clip_range* is not a ``(low, high)`` pair with
                ``low < high``.
        """
        self.training_mode = training_mode
        if clip_range is not None:
            if len(clip_range) != 2:
                raise ValueError(
                    f"`clip_range` must be a (low, high) pair, got {clip_range!r}."
                )
            low, high = float(clip_range[0]), float(clip_range[1])
            if not low < high:
                raise ValueError(
                    f"`clip_range` must have low < high, got ({low}, {high})."
                )
            clip_range = (low, high)
        self.clip_range = clip_range
        self.normalize = Normalize(means, stds)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Run preprocessing on a sample.

        Args:
            sample (dict[str, Any]): The sample to preprocess.

        Returns:
            dict[str, Any]: The preprocessed sample.
        """
        if "image" in sample:
            image = self.normalize(sample["image"])
            if self.clip_range is not None:
                image = torch.clip(image, self.clip_range[0], self.clip_range[1])
            sample["image"] = image
        if "mask" in sample:
            sample["mask"] = sample["mask"].squeeze().long()

        # We remove the bounding box when training as it causes problems in lightning
        # based trainers
        if self.training_mode and "bbox" in sample:  # for torchgeo < 0.6
            del sample["bbox"]
        if self.training_mode and "bounds" in sample:  # for torchgeo >= 0.6
            del sample["bounds"]

        if not self.training_mode and "bbox" in sample:  # for torchgeo >= 0.6
            sample["bounds"] = sample["bbox"]

        return sample
