# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for creating visualization of building footprints."""

import argparse
import os

import fiona
import numpy as np
import rasterio
import rasterio.enums
import rasterio.features
import shapely.geometry
from tqdm import tqdm


IDX_TO_COLOR = np.array(
    [
        [0, 0, 0, 0],
        [255, 255, 255, 255],
        [252, 190, 165, 255],
        [251, 112, 80, 255],
        [211, 32, 32, 255],
        [103, 0, 13, 255],
    ],
    dtype=np.uint8,
)


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged_footprints_fn",
        type=str,
        required=True,
        help="Path to the footprint file",
    )
    parser.add_argument(
        "--predictions_fn", type=str, required=True, help="Path to the prediction file"
    )
    parser.add_argument(
        "--output_fn", type=str, required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists",
    )

    return parser


def classify(x):
    thresholds = [0.2, 0.4, 0.6, 0.8]
    for i, threshold in enumerate(thresholds):
        if x <= threshold:
            return i
    return len(thresholds)


def main(args):
    """Main function for the merge_with_building_footprints.py script."""
    if os.path.exists(args.output_fn) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_fn} already exists. Use --overwrite to overwrite it."
        )

    with rasterio.open(args.predictions_fn, "r") as src:
        predictions_crs = src.crs.to_string()
        height, width = src.shape
        transform = src.transform
        profile = src.profile

    with fiona.open(args.merged_footprints_fn, "r") as src:
        footprints_crs = src.crs.to_string()

    assert footprints_crs == predictions_crs

    ############################################
    # Read predictions within building footprints
    # and track damage values
    ############################################
    shape_vals = []
    with fiona.open(args.merged_footprints_fn) as f:
        for row in f:
            geom = row["geometry"]
            val = classify(row["properties"]["damage_pct_0m"]) + 1
            shape_vals.append((geom, val))

    mask = rasterio.features.rasterize(
        shape_vals,
        out_shape=(height, width),
        transform=transform,
        fill=0,
    )

    colors = IDX_TO_COLOR[mask]
    colors = colors.transpose(2, 0, 1)

    profile["count"] = 4
    profile["nodata"] = 0

    with rasterio.open(args.output_fn, "w", **profile) as f:
        f.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]
        f.write(colors)


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
