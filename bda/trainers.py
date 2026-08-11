# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom torchgeo trainers."""

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torchgeo.trainers import SemanticSegmentationTask
import kornia.augmentation as K
import torch.nn as nn
import segmentation_models_pytorch as smp


def _supervised_logits(y_hat: Tensor, no_damage_index: int) -> Tensor:
    """Return only the logits for classes that receive supervision.

    Channel 0 is retained in the model output as the "Unlabeled" channel but is
    excluded here so that it never receives a gradient. "No Damage" is a weak
    mask annotation rather than a class the model predicts, so it is the final
    mask value and has no output channel of its own.
    """
    num_channels = y_hat.shape[1]
    if no_damage_index != num_channels:
        raise ValueError(
            f"no_damage_index ({no_damage_index}) must equal the number of output "
            f"channels ({num_channels}); 'No Damage' is a weak label that must be "
            "the final mask value and carry no output channel of its own."
        )
    return y_hat[:, 1:]


def constraint_segmentation_loss_components(
    y_hat: Tensor,
    y: Tensor,
    no_damage_index: int,
    damaged_class_index: int = 3,
) -> tuple[Tensor, Tensor]:
    """Compute CE and weak-label constraint losses separately.

    Standard cross-entropy is applied to every labeled pixel *except* those of
    the "No Damage" class. At a "No Damage" pixel we don't know the true class
    (building vs. background), only that the building is *not* damaged -- so
    instead of a hard label we add a penalty on the predicted probability of the
    "Damaged Building" class there.

    Args:
        y_hat: Predicted logits of shape ``(N, C, H, W)``. Channel 0 is the
            non-deployable "Unlabeled" output and receives no gradient.
        y: Integer mask of shape ``(N, H, W)``; 0 is the unlabeled/ignored class.
        no_damage_index: Mask value of the "No Damage" class. This equals
            ``labels.classes.index("No Damage") + 1`` and therefore depends on
            the config (4 without a "Cloud" class, 5 with one), so it must be
            passed in rather than hardcoded.
        damaged_class_index: Output channel / mask value of the "Damaged
            Building" class that is penalized at "No Damage" pixels.

    Returns:
        The ``(cross_entropy_loss, constraint_loss)`` pair.
    """
    supervised_logits = _supervised_logits(y_hat, no_damage_index)

    # Supervised logits are zero-indexed, while deployable mask values start at
    # 1. Unlabeled and No Damage pixels use a dedicated ignored target.
    ce_targets = torch.full_like(y, -100)
    standard_mask = (y > 0) & (y != no_damage_index)
    ce_targets[standard_mask] = y[standard_mask] - 1

    ce_loss = F.cross_entropy(
        supervised_logits, ce_targets, ignore_index=-100, reduction="none"
    )
    if standard_mask.any():
        ce_loss = ce_loss[standard_mask].mean()
    else:
        ce_loss = supervised_logits.sum() * 0.0

    constraint_mask = y == no_damage_index
    if constraint_mask.any():
        if not 0 < damaged_class_index < no_damage_index:
            raise ValueError(
                "damaged_class_index must identify a supervised mask class before "
                "no_damage_index"
            )
        probs = F.softmax(supervised_logits, dim=1)
        constraint_loss = probs[:, damaged_class_index - 1, :, :][
            constraint_mask
        ].mean()
    else:
        constraint_loss = supervised_logits.sum() * 0.0

    return ce_loss, constraint_loss


def constraint_segmentation_loss(
    y_hat: Tensor,
    y: Tensor,
    no_damage_index: int,
    damaged_class_index: int = 3,
) -> Tensor:
    """Return the combined CE and weak-label constraint loss."""
    ce_loss, constraint_loss = constraint_segmentation_loss_components(
        y_hat, y, no_damage_index, damaged_class_index
    )

    return ce_loss + constraint_loss


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    """A custom trainer for semantic segmentation tasks."""

    def __init__(
        self,
        *args,
        use_constraint_loss=False,
        no_damage_index=None,
        damaged_class_index=3,
        **kwargs,
    ):
        if "ignore" in kwargs:
            del kwargs[
                "ignore"
            ]  # workaround for https://github.com/microsoft/torchgeo/pull/2314, can be removed with torchgeo 0.7

        super().__init__(*args, **kwargs)

        self.use_constraint_loss = use_constraint_loss
        # Mask value of the "No Damage" class for the constraint loss. Mask values
        # are (index in labels.classes) + 1, so this is config-dependent and is
        # passed in by fine_tune.py rather than hardcoded.
        self.no_damage_index = no_damage_index
        # Output channel / mask value of the "Damaged Building" class penalized at
        # "No Damage" pixels (consistently 3 in this repo's schema).
        self.damaged_class_index = damaged_class_index

        self.train_augs = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

    def _constraint_metric_inputs(
        self, y_hat: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Prepare predictions and targets for metrics under weak supervision."""
        metric_targets = y.clone()
        metric_targets[metric_targets == self.no_damage_index] = 0
        return y_hat, metric_targets

    def _constraint_losses(
        self, y_hat: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute and return CE, constraint, and total losses."""
        if self.no_damage_index is None:
            raise ValueError(
                "use_constraint_loss is True but no_damage_index is not set. "
                "Pass the mask value of the 'No Damage' class (see fine_tune.py)."
            )
        ce_loss, constraint_loss = constraint_segmentation_loss_components(
            y_hat, y, self.no_damage_index, self.damaged_class_index
        )
        return ce_loss, constraint_loss, ce_loss + constraint_loss

    def configure_callbacks(self) -> list[Callback]:
        """Configures the callbacks for the trainer.

        Returns:
            an empty list to override the default callbacks, we set these in the Trainer
        """
        return []

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams['loss']
        ignore_index = self.hparams['ignore_index']
        if loss == 'ce':
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=self.hparams['class_weights']
            )
        elif loss == 'dice':
            self.criterion = smp.losses.DiceLoss(
                mode='multiclass',
                ignore_index=ignore_index,
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        batch = self.train_augs(batch)
        x = batch["image"]
        y = batch["mask"]

        batch_size = x.shape[0]
        y_hat = self(x)

        if self.use_constraint_loss:
            ce_loss, constraint_loss, loss = self._constraint_losses(y_hat, y)
            metric_logits, metric_targets = self._constraint_metric_inputs(y_hat, y)
            self.log("train_ce_loss", ce_loss, batch_size=batch_size)
            self.log(
                "train_constraint_loss", constraint_loss, batch_size=batch_size
            )
        else:
            loss = self.criterion(y_hat, y)
            metric_logits, metric_targets = y_hat, y

        self.log("train_loss", loss, batch_size=batch_size)
        self.train_metrics(metric_logits, metric_targets)
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute validation loss using the same weak-label semantics as training."""
        if not self.use_constraint_loss:
            return super().validation_step(batch, batch_idx, dataloader_idx)
        if self.no_damage_index is None:
            raise ValueError(
                "use_constraint_loss is True but no_damage_index is not set"
            )

        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x)
        ce_loss, constraint_loss, loss = self._constraint_losses(y_hat, y)
        metric_logits, metric_targets = self._constraint_metric_inputs(y_hat, y)
        self.val_metrics(metric_logits, metric_targets)
        self.log("val_ce_loss", ce_loss, batch_size=batch_size)
        self.log("val_constraint_loss", constraint_loss, batch_size=batch_size)
        self.log("val_loss", loss, batch_size=batch_size)

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute test loss using the same weak-label semantics as training."""
        if not self.use_constraint_loss:
            return super().test_step(batch, batch_idx, dataloader_idx)
        if self.no_damage_index is None:
            raise ValueError(
                "use_constraint_loss is True but no_damage_index is not set"
            )

        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y_hat = self(x)
        ce_loss, constraint_loss, loss = self._constraint_losses(y_hat, y)
        metric_logits, metric_targets = self._constraint_metric_inputs(y_hat, y)
        self.test_metrics(metric_logits, metric_targets)
        self.log("test_ce_loss", ce_loss, batch_size=batch_size)
        self.log("test_constraint_loss", constraint_loss, batch_size=batch_size)
        self.log("test_loss", loss, batch_size=batch_size)
