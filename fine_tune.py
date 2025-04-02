# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for fine-tuning a segmentation model to detect damage in satellite imagery."""

import argparse
import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from bda.config import get_args
from bda.datamodules import SegmentationDataModule
from bda.trainers import CustomSemanticSegmentationTask


def add_fine_tune_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds the arguments for the fine_tune.py script to the base parser."""
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="Directory that contains an `images/` and `masks/` directory",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment (used for TensorBoard logging)",
    )
    parser.add_argument("--training.gpu_id", type=int, help="GPU id to use")
    parser.add_argument("--training.batch_size", type=int, help="Batch size")
    parser.add_argument(
        "--training.learning_rate", type=float, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--training.max_epochs", type=int, help="Maximum number of epochs to train for"
    )
    parser.add_argument(
        "--training.log_dir", type=str, help="Directory to write TensorBoard logs to"
    )
    parser.add_argument(
        "--training.checkpoint_subdir",
        type=str,
        help="Directory to write model checkpoints to",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output dataset if it already exists",
    )
    # NOTE: we don't include `--labels.classes` here because we assume that the classes
    # are the same as the ones used to create the masks.
    # NOTE: we don't include `--imagery.num_channels`, `--imagery.normalization_means`,
    # or `--imagery.normalization_stds` here either because we assume that you won't
    # want to change that

    return parser


def main() -> None:
    """Main function for the fine_tune.py script."""
    args = get_args(description=__doc__, add_extra_parser=add_fine_tune_parser)

    experiment_dir = args["experiment_dir"]
    assert os.path.exists(os.path.join(experiment_dir, "images/"))
    assert os.path.exists(os.path.join(experiment_dir, "masks/"))

    checkpoint_dir = os.path.join(experiment_dir, args["training"]["checkpoint_subdir"])
    if os.path.exists(checkpoint_dir) and not args["overwrite"]:
        print(
            "Experiment output files already exist, use --overwrite to overwrite them."
            + " Exiting."
        )
        return
    os.makedirs(checkpoint_dir, exist_ok=True)

    datamodule = SegmentationDataModule(
        os.path.join(experiment_dir, "images/"),
        os.path.join(experiment_dir, "masks/"),
        batch_size=args["training"]["batch_size"],
        num_workers=24,
        train_batches_per_epoch=1024,
        means=args["imagery"]["normalization_means"],
        stds=args["imagery"]["normalization_stds"],
    )

    # we include +1 to account for our 0 "not labeled" class
    num_classes = len(args["labels"]["classes"]) + 1
    task = CustomSemanticSegmentationTask(
        model="unet",
        backbone="resnext50_32x4d",
        weights=True,  # use ImageNet pre-trained weights
        in_channels=args["imagery"]["num_channels"],
        num_classes=num_classes,
        loss="ce",
        ignore_index=0,  # we use 0 as a "not labeled" class by convention
        lr=args["training"]["learning_rate"],
        patience=10,
        use_constraint_loss=args["training"]["use_constraint_loss"],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=checkpoint_dir, save_top_k=2, save_last=True
    )

    tb_logger = TensorBoardLogger(
        save_dir=args["training"]["log_dir"], name=args["experiment_name"]
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=[tb_logger],
        min_epochs=10,
        max_epochs=args["training"]["max_epochs"],
        accelerator="gpu",
        devices=[args["training"]["gpu_id"]],
    )

    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    main()
