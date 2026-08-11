# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the configurable imagery clipping in ``bda.preprocess``."""

import pytest
import torch

from bda.config import resolve_clip_range
from bda.preprocess import Preprocessor


def _sample():
    # Values chosen so that, with mean 100 / std 10, the normalized results are
    # -2, 0 and 5 -- i.e. below, inside and above the default (0, 1) range.
    return {"image": torch.tensor([[[80.0, 100.0, 150.0]]])}


def _normalized(clip_range):
    pre = Preprocessor(
        training_mode=True, means=[100.0], stds=[10.0], clip_range=clip_range
    )
    return pre(_sample())["image"].flatten().tolist()


def test_default_clips_to_zero_one():
    """The default preserves the behavior from before this was configurable."""
    assert _normalized((0, 1)) == [0.0, 0.0, 1.0]


def test_symmetric_range_keeps_negative_values():
    """A (-3, 3) range is what standardized imagery needs."""
    assert _normalized((-3, 3)) == [-2.0, 0.0, 3.0]


def test_none_disables_clipping():
    assert _normalized(None) == [-2.0, 0.0, 5.0]


@pytest.mark.parametrize("bad", [(1, 0), (2, 2), (5,), (0, 1, 2)])
def test_invalid_clip_ranges_are_rejected(bad):
    with pytest.raises(ValueError):
        Preprocessor(training_mode=True, means=[0.0], stds=[1.0], clip_range=bad)


def test_resolve_clip_range_defaults_to_zero_one():
    """A config without the key keeps the historical behavior."""
    assert resolve_clip_range({}) == (0.0, 1.0)


def test_resolve_clip_range_explicit_null_disables():
    assert resolve_clip_range({"clip_range": None}) is None


def test_resolve_clip_range_no_clip_takes_precedence():
    assert resolve_clip_range({"clip_range": [0, 1], "no_clip": True}) is None


def test_resolve_clip_range_parses_values():
    assert resolve_clip_range({"clip_range": [-3, 3]}) == (-3.0, 3.0)
