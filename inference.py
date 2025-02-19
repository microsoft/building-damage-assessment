# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for running inference on a satellite image with a torchgeo segmentation model."""

import argparse
import math
import os
import time

import numpy as np
import rasterio
import torch
import tqdm
from rasterio.enums import ColorInterp
from torch.utils.data import DataLoader

from bda.config import get_args
from bda.datasets import TileDataset, stack_samples
from bda.samplers import GridGeoSampler
from bda.preprocess import Preprocessor
from bda.trainers import CustomSemanticSegmentationTask


def add_inference_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds the arguments for the inference.py script to the base parser."""
    parser.add_argument(
        "--inference.checkpoint_fn",
        type=str,
        help="Model checkpoint to load, defaults to `last.ckpt` in the training checkpoints directory",
    )
    parser.add_argument(
        "--inference.output_subdir",
        type=str,
        help="Subdirectory to save outputs in, defaults to `outputs/` in the experiment directory",
    )
    parser.add_argument("--inference.gpu_id", type=int, help="GPU id to use")
    parser.add_argument(
        "--inference.patch_size", type=int, help="Size of patch to use for inference"
    )
    parser.add_argument("--inference.batch_size", type=int, help="Batch size")
    parser.add_argument(
        "--inference.padding",
        type=int,
        help="Number of pixels to throw away from each side of the patch after inference",
    )  # TODO: better description
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrites the outputs if they exist"
    )
    # NOTE: we don't include some flags like `--imagery.normalization_means` or
    # `--imagery.normalization_stds` here because we assume that you won't want to
    # change them

    return parser


def main() -> None:
    """Main function for the inference.py script."""
    args = get_args(description=__doc__, add_extra_parser=add_inference_parser)

    input_model_checkpoint = os.path.join(
        args["experiment_dir"], args["inference"]["checkpoint_fn"]
    )
    print(input_model_checkpoint)
    input_image_fn = args["imagery"]["raw_fn"]
    patch_size = args["inference"]["patch_size"]
    padding = args["inference"]["padding"]
    output_dir = os.path.join(
        args["experiment_dir"], args["inference"]["output_subdir"]
    )

    # Sanity checks
    assert os.path.exists(input_model_checkpoint)
    assert input_model_checkpoint.endswith(".ckpt")
    assert os.path.exists(input_image_fn)
    assert input_image_fn.endswith(".tif") or input_image_fn.endswith(".vrt")
    assert int(math.log(patch_size, 2)) == math.log(patch_size, 2)
    stride = patch_size - padding * 2

    image_name = os.path.basename(input_image_fn).replace(".tif", "")
    output_fn = os.path.join(output_dir, f"{image_name}_predictions.tif")
    if os.path.exists(output_fn) and not args["overwrite"]:
        print(
            "Experiment output files already exist, use --overwrite to overwrite them."
            + " Exiting."
        )
        return
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{args['inference']['gpu_id']}" if torch.cuda.is_available() else "cpu"
    )

    # Load task and data
    tic = time.time()
    task = CustomSemanticSegmentationTask.load_from_checkpoint(input_model_checkpoint, map_location="cpu")
    task.freeze()
    model = task.model
    model = model.eval().to(device)

    preprocess = Preprocessor(
        training_mode=False,
        means=args["imagery"]["normalization_means"],
        stds=args["imagery"]["normalization_stds"],
    )

    dataset = TileDataset([[input_image_fn]], mask_fns=None, transforms=preprocess)
    sampler = GridGeoSampler([[input_image_fn]], [0], patch_size=patch_size, stride=stride)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args["inference"]["batch_size"],
        num_workers=12,
        collate_fn=stack_samples,
    )

    # Run inference
    tic = time.time()
    with rasterio.open(input_image_fn) as f:
        input_height, input_width = f.shape
        profile = f.profile

    print(f"Input size: {input_height} x {input_width}")
    assert patch_size <= input_height
    assert patch_size <= input_width
    output = np.zeros((input_height, input_width), dtype=np.uint8)

    # NOTE: we can make output quiet by adding a flag to set `dl_enumerator = dataloader`
    dl_enumerator = tqdm.tqdm(dataloader)

    for batch in dl_enumerator:
        images = batch["image"].to(device)
        x_coords = batch["x"]
        y_coords = batch["y"]
        batch_size = images.shape[0]
        with torch.inference_mode():
            predictions = task(images)
            predictions = predictions.argmax(axis=1).cpu().numpy().astype(np.uint8)

        for i in range(batch_size):
            height, width = predictions[i].shape
            y = int(y_coords[i])
            x = int(x_coords[i])
            output[y+padding:y+height-padding, x+padding:x+width-padding] = predictions[i][padding:-padding, padding:-padding]

    print(f"Finished running model in {time.time()-tic:0.2f} seconds")

    # Save predictions
    tic = time.time()
    profile["driver"] = "GTiff"
    profile["count"] = 1
    profile["dtype"] = "uint8"
    profile["compress"] = "lzw"
    profile["predictor"] = 2
    profile["nodata"] = 0
    profile["blockxsize"] = 512
    profile["blockysize"] = 512
    profile["tiled"] = True
    profile["interleave"] = "pixel"

    with rasterio.open(output_fn, "w", **profile) as f:
        f.write(output, 1)
        f.write_colormap(
            1,
            {
                1: (
                    0,
                    0,
                    0,
                    0,
                ),  # this alpha doesn't work because of a limitation in TIFFs
                2: (0, 255, 0, 255),
                3: (255, 0, 0, 255),
            },
        )
        f.colorinterp = [ColorInterp.palette]

    print(f"Finished saving predictions in {time.time()-tic:0.2f} seconds")


if __name__ == "__main__":
    main()
