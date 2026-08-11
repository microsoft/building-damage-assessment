# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for fine-tuning a segmentation model to detect damage in satellite imagery."""

import argparse
import glob
import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from bda.config import (
    add_clip_range_args,
    get_args,
    normalize_gpu_ids,
    resolve_clip_range,
)
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
    parser.add_argument(
        "--training.gpu_ids",
        type=int,
        nargs="+",
        help=(
            "One or more GPU ids for multi-GPU (DDP) training, e.g."
            " `--training.gpu_ids 0 1 2 3`. Overrides `--training.gpu_id`."
        ),
    )
    parser.add_argument("--training.batch_size", type=int, help="Batch size")
    parser.add_argument(
        "--training.preload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to read all tiles into memory instead of reading each patch"
        + " from disk (much faster, but requires the tiles to fit in RAM)",
    )
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

    parser = add_clip_range_args(parser)

    return parser


def main() -> None:
    """Main function for the fine_tune.py script."""
    args = get_args(description=__doc__, add_extra_parser=add_fine_tune_parser)

    # Enables TF32 tensor cores for fp32 matmuls (e.g. on A100/H100 GPUs)
    torch.set_float32_matmul_precision("high")

    experiment_dir = args["experiment_dir"]
    assert os.path.exists(os.path.join(experiment_dir, "images/"))
    assert os.path.exists(os.path.join(experiment_dir, "masks/"))

    checkpoint_dir = os.path.join(experiment_dir, args["training"]["checkpoint_subdir"])
    # Check for existing *checkpoints*, not just the directory. Under DDP,
    # Lightning re-runs this script in a subprocess per GPU; the main process
    # creates the (empty) checkpoint directory before the workers start, so a
    # bare directory-existence check would make every worker exit early and the
    # run would hang.
    existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if existing_checkpoints and not args["overwrite"]:
        print(
            "Experiment output files already exist, use --overwrite to overwrite them."
            + " Exiting."
        )
        return
    os.makedirs(checkpoint_dir, exist_ok=True)

    gpu_ids = normalize_gpu_ids(
        args["training"].get("gpu_ids"), args["training"].get("gpu_id")
    )
    world_size = max(1, len(gpu_ids))

    # `train_batches_per_epoch` is per-process under DDP, so divide it by the
    # number of GPUs. This keeps the total batches seen per epoch (and thus the
    # epoch wall-clock time) roughly constant as GPUs are added -- i.e. more GPUs
    # makes each epoch faster instead of just processing proportionally more data.
    train_batches_per_epoch = max(1, 1024 // world_size)

    datamodule = SegmentationDataModule(
        os.path.join(experiment_dir, "images/"),
        os.path.join(experiment_dir, "masks/"),
        batch_size=args["training"]["batch_size"],
        num_workers=4,
        train_batches_per_epoch=train_batches_per_epoch,
        means=args["imagery"]["normalization_means"],
        stds=args["imagery"]["normalization_stds"],
        preload=args["training"].get("preload", True),
        clip_range=resolve_clip_range(args["imagery"]),
    )

    classes = args["labels"]["classes"]

    # The constraint loss penalizes the predicted "Damaged Building" probability
    # at "No Damage" pixels. Mask values are (index in classes) + 1, so the
    # "No Damage" value is config-dependent (4 without a "Cloud" class, 5 with
    # one) and must be derived here rather than hardcoded in the trainer.
    use_constraint_loss = args["training"]["use_constraint_loss"]
    no_damage_index = None
    damaged_class_index = 3
    if use_constraint_loss:
        missing = [c for c in ("No Damage", "Damaged Building") if c not in classes]
        if missing:
            raise ValueError(
                "training.use_constraint_loss is true but labels.classes is "
                f"missing {missing}. The constraint loss penalizes 'Damaged "
                "Building' probability at 'No Damage' pixels, so both classes "
                "are required."
            )
        no_damage_index = classes.index("No Damage") + 1
        damaged_class_index = classes.index("Damaged Building") + 1
        if no_damage_index != len(classes):
            raise ValueError(
                "'No Damage' must be the final labels.classes entry when "
                "training.use_constraint_loss is true. It is a weak label, not "
                "a deployable output class."
            )
        print(
            f"Constraint loss enabled: penalizing P(Damaged Building="
            f"{damaged_class_index}) at No Damage (={no_damage_index}) pixels"
        )

    # Include output channel 0 for "not labeled". Under constraint training,
    # "No Damage" is only a weak mask annotation and gets no output channel.
    num_classes = len(classes) if use_constraint_loss else len(classes) + 1

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
        use_constraint_loss=use_constraint_loss,
        no_damage_index=no_damage_index,
        damaged_class_index=damaged_class_index,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=checkpoint_dir, save_top_k=2, save_last=True
    )

    os.makedirs(args["training"]["log_dir"], exist_ok=True)
    tb_logger = TensorBoardLogger(
        save_dir=args["training"]["log_dir"], name=args["experiment_name"]
    )

    # More than one GPU -> data-parallel training via DDP (Lightning's robust
    # multi-GPU strategy). `use_distributed_sampler=False` keeps our custom
    # RandomGeoSampler instead of Lightning injecting a DistributedSampler.
    strategy = "ddp" if world_size > 1 else "auto"
    print(
        f"Training on GPU id(s): {gpu_ids} (strategy={strategy}, "
        f"{train_batches_per_epoch} train batches/epoch/process)"
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=[tb_logger],
        min_epochs=10,
        max_epochs=args["training"]["max_epochs"],
        accelerator="gpu",
        devices=gpu_ids if gpu_ids else "auto",
        strategy=strategy,
        use_distributed_sampler=False,
        precision="bf16-mixed",
    )

    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    main()
