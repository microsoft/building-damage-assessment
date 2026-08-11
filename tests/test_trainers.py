# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the No Damage constraint loss in ``bda.trainers``."""

import torch
import torch.nn.functional as F

from bda.trainers import constraint_segmentation_loss


def _reference_loss(y_hat, y, no_damage_index, damaged_class_index):
    """Independent re-implementation used to check the helper."""
    ce = F.cross_entropy(y_hat, y, ignore_index=0, reduction="none")
    standard_mask = (y > 0) & (y != no_damage_index)
    loss = ce[standard_mask].mean()
    constraint_mask = y == no_damage_index
    if constraint_mask.any():
        probs = F.softmax(y_hat, dim=1)
        loss = loss + probs[:, damaged_class_index, :, :][constraint_mask].mean()
    return loss


def test_constraint_fires_for_4_class_layout():
    """No Damage == 4 (config without a 'Cloud' class)."""
    torch.manual_seed(0)
    # num_classes = 5 -> channels 0..4; mask values 0..4, No Damage = 4
    y_hat = torch.randn(1, 5, 4, 4)
    y = torch.tensor([[[0, 1, 2, 3], [1, 2, 3, 4], [4, 4, 1, 2], [3, 2, 1, 4]]])

    got = constraint_segmentation_loss(y_hat, y, no_damage_index=4, damaged_class_index=3)
    expected = _reference_loss(y_hat, y, 4, 3)
    assert torch.allclose(got, expected)


def test_constraint_fires_for_5_class_layout():
    """No Damage == 5 (config with a 'Cloud' class)."""
    torch.manual_seed(1)
    # num_classes = 6 -> channels 0..5; mask values 0..5, No Damage = 5
    y_hat = torch.randn(1, 6, 4, 4)
    y = torch.tensor([[[0, 1, 2, 3], [4, 5, 3, 2], [5, 5, 1, 2], [3, 4, 1, 5]]])

    got = constraint_segmentation_loss(y_hat, y, no_damage_index=5, damaged_class_index=3)
    expected = _reference_loss(y_hat, y, 5, 3)
    assert torch.allclose(got, expected)


def test_penalty_increases_with_predicted_damage_at_no_damage_pixels():
    """Higher P(Damaged Building) at No Damage pixels must raise the loss."""
    # One standard Background pixel (1) and one No Damage pixel (4).
    y = torch.tensor([[[1, 4]]])  # shape (1, 1, 2)

    low = torch.zeros(1, 5, 1, 2)
    low[0, 1, 0, 0] = 10.0   # confidently Background at the standard pixel
    low[0, 1, 0, 1] = 10.0   # low P(Damaged Building) at the No Damage pixel

    high = low.clone()
    high[0, 1, 0, 1] = 0.0
    high[0, 3, 0, 1] = 10.0  # high P(Damaged Building) at the No Damage pixel

    loss_low = constraint_segmentation_loss(low, y, no_damage_index=4)
    loss_high = constraint_segmentation_loss(high, y, no_damage_index=4)
    assert loss_high > loss_low


def test_no_constraint_pixels_equals_plain_ce():
    """With no No Damage pixels the loss is just CE over labeled pixels."""
    torch.manual_seed(2)
    y_hat = torch.randn(1, 5, 3, 3)
    y = torch.tensor([[[0, 1, 2], [3, 1, 2], [2, 3, 1]]])  # no value 4

    got = constraint_segmentation_loss(y_hat, y, no_damage_index=4)
    ce = F.cross_entropy(y_hat, y, ignore_index=0, reduction="none")
    expected = ce[(y > 0) & (y != 4)].mean()
    assert torch.allclose(got, expected)


def test_all_no_damage_patch_is_finite():
    """A patch that is only No Damage (+unlabeled) must not produce NaN."""
    torch.manual_seed(4)
    y_hat = torch.randn(1, 5, 3, 3, requires_grad=True)
    y = torch.tensor([[[0, 4, 4], [4, 0, 4], [4, 4, 0]]])  # only 0 and 4

    loss = constraint_segmentation_loss(y_hat, y, no_damage_index=4)
    assert torch.isfinite(loss)
    # Loss is purely the constraint penalty here and should still backprop.
    loss.backward()
    assert y_hat.grad is not None


def test_fully_unlabeled_patch_is_finite_zero():
    """A patch with no labeled pixels at all yields a finite (zero) loss."""
    y_hat = torch.randn(1, 5, 2, 2)
    y = torch.zeros(1, 2, 2, dtype=torch.long)  # all unlabeled
    loss = constraint_segmentation_loss(y_hat, y, no_damage_index=4)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0

