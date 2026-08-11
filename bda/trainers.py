# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom torchgeo trainers."""

from typing import Any
from torch import Tensor
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torchgeo.trainers import SemanticSegmentationTask
import kornia.augmentation as K
import torch.nn as nn
import segmentation_models_pytorch as smp


def constraint_segmentation_loss(
    y_hat: Tensor,
    y: Tensor,
    no_damage_index: int,
    damaged_class_index: int = 3,
) -> Tensor:
    """Constraint loss for weakly-supervised "No Damage" labels.

    Standard cross-entropy is applied to every labeled pixel *except* those of
    the "No Damage" class. At a "No Damage" pixel we don't know the true class
    (building vs. background), only that the building is *not* damaged -- so
    instead of a hard label we add a penalty on the predicted probability of the
    "Damaged Building" class there.

    Args:
        y_hat: Predicted logits of shape ``(N, C, H, W)``.
        y: Integer mask of shape ``(N, H, W)``; 0 is the unlabeled/ignored class.
        no_damage_index: Mask value of the "No Damage" class. This equals
            ``labels.classes.index("No Damage") + 1`` and therefore depends on
            the config (4 without a "Cloud" class, 5 with one), so it must be
            passed in rather than hardcoded.
        damaged_class_index: Output channel / mask value of the "Damaged
            Building" class that is penalized at "No Damage" pixels.

    Returns:
        The scalar loss.
    """
    ce_loss = F.cross_entropy(y_hat, y, ignore_index=0, reduction="none")
    standard_mask = (y > 0) & (y != no_damage_index)
    if standard_mask.any():
        loss = ce_loss[standard_mask].mean()
    else:
        # No "standard" labeled pixels in this batch (e.g. a patch that is
        # entirely "No Damage" + unlabeled). Avoid a NaN from mean() over an
        # empty tensor while staying attached to the autograd graph.
        loss = y_hat.sum() * 0.0

    constraint_mask = y == no_damage_index
    if constraint_mask.any():
        probs = F.softmax(y_hat, dim=1)
        penalty = probs[:, damaged_class_index, :, :][constraint_mask]
        loss = loss + penalty.mean()

    return loss


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
            if self.no_damage_index is None:
                raise ValueError(
                    "use_constraint_loss is True but no_damage_index is not set. "
                    "Pass the mask value of the 'No Damage' class (see fine_tune.py)."
                )
            loss = constraint_segmentation_loss(
                y_hat, y, self.no_damage_index, self.damaged_class_index
            )
        else:
            loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, batch_size=batch_size)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size)
        return loss
