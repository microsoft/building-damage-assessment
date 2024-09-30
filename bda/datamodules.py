# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom lightning datamodules."""

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.transforms import AugmentationSequential

from .preprocess import Preprocessor


class SegmentationDataModule(LightningDataModule):
    """Lightning datamodule for segmentation tasks."""

    def __init__(
        self,
        image_fn_root,
        mask_fn_root,
        batch_size=64,
        patch_size=256,
        num_workers=6,
        train_batches_per_epoch=512,
        val_batches_per_epoch=32,
        means=[0, 0, 0, 0],
        stds=[500, 500, 500, 500],
    ):
        """Initialize the SegmentationDataModule class."""
        super().__init__()

        self.image_fn_root = image_fn_root
        self.mask_fn_root = mask_fn_root
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch

        self.means = means
        self.stds = stds

        self.preprocess = Preprocessor(training_mode=True, means=means, stds=stds)

        self.train_aug = AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            data_keys=["image", "mask"],
        )

    def setup(self, stage=None):
        """Set up the datasets."""
        print(f"setting up {stage}")
        self.image_ds = RasterDataset(self.image_fn_root, transforms=self.preprocess)

        self.mask_ds = RasterDataset(self.mask_fn_root, transforms=self.preprocess)
        self.mask_ds.is_image = False

        self.ds = self.image_ds & self.mask_ds

    def train_dataloader(self):
        """Set up the train dataloader."""
        sampler = RandomBatchGeoSampler(
            self.ds,
            size=self.patch_size,
            batch_size=self.batch_size,
            length=self.train_batches_per_epoch * self.batch_size,
        )

        return DataLoader(
            self.ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def val_dataloader(self):
        """Set up the val dataloader."""
        sampler = RandomBatchGeoSampler(
            self.ds,
            size=self.patch_size,
            batch_size=self.batch_size,
            length=self.val_batches_per_epoch * self.batch_size,
        )

        return DataLoader(
            self.ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def plot(self, sample):
        """Plot a single sample from the dataset."""
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        mask = sample["mask"].squeeze(0)
        ncols = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0)
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

        axs[0].imshow(image[:, :, :3])
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=3, interpolation="none")
        axs[1].axis("off")

        if showing_predictions:
            axs[2].imshow(pred, vmin=0, vmax=3, interpolation="none")
            axs[2].axis("off")

        return fig

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Runs augmentation when in training mode."""
        if self.trainer:
            if self.trainer.training:
                batch = self.train_aug(batch)
        return batch
