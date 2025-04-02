# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom torchgeo trainers."""

from typing import Any
from torch import Tensor
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torchgeo.trainers import SemanticSegmentationTask
import kornia.augmentation as K


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    """A custom trainer for semantic segmentation tasks."""

    def __init__(self, *args, use_constraint_loss=False, **kwargs):
        if "ignore" in kwargs:
            del kwargs[
                "ignore"
            ]  # workaround for https://github.com/microsoft/torchgeo/pull/2314, can be removed with torchgeo 0.7
        super().__init__(*args, **kwargs)

        self.use_constraint_loss = use_constraint_loss

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
            ce_loss = F.cross_entropy(y_hat, y, ignore_index=0, reduction="none")
            standard_mask = (y > 0) & (y < 4)
            loss = ce_loss[standard_mask].mean()

            constraint_mask = y == 4
            if constraint_mask.any():
                probs = F.softmax(y_hat, dim=1)
                penalty = probs[:, 3, :, :][constraint_mask]
                constraint_loss = penalty.mean()
                loss = loss + constraint_loss
        else:
            loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, batch_size=batch_size)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size)
        return loss
