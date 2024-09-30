# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for extracting a raw DPM from a color mapped DPM.

The EOS DPMs are provided only as colormapped GeoTIFFs. This script extracts a raw
DPM from the colormapped DPM (approximately by guessing at the colormap and assuming the
mapping is linear). Use the `--visualize_cmap` flag to visualize the extracted colormap.

From the EOS docs we know that the colormap is _not_ linear, so this is temporary
solution.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from tqdm import tqdm


def main():
    """Main function for the convert_colormapped_dpm_to_raw.py script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_fn", type=str, help="Path to input color mapped DPM file"
    )
    parser.add_argument("--output_fn", type=str, help="Path to output raw file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    parser.add_argument(
        "--visualize_cmap",
        action="store_true",
        help="Visualize the extracted color map",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_fn):
        print(f"ERROR: Input file does not exist: {args.input_fn}")
        sys.exit(1)

    if os.path.exists(args.output_fn) and not args.overwrite:
        print(f"ERROR: Output file already exists: {args.output_fn}, use `--overwrite`")
        sys.exit(1)
    elif os.path.exists(args.output_fn) and args.overwrite:
        print(f"WARNING: Overwriting output file: {args.output_fn}")

    output_dir = os.path.dirname(args.output_fn)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Read file
    print("Reading input file...")
    with rasterio.open(args.input_fn) as f:
        dpm = f.read()
        profile = f.profile
    num_channels, height, width = dpm.shape
    assert num_channels == 4

    # Extract the colormap from the DPM
    print("Extracting the colormap...")
    dpm_uniques = dpm.copy().reshape(4, dpm.shape[1] * dpm.shape[2]).T
    vals = np.unique(dpm_uniques, axis=0)
    sorted_colors = vals[np.lexsort((-vals[:, 3],))]
    lookup = {
        tuple(color): 1 - (i / sorted_colors.shape[0])
        for i, color in enumerate(sorted_colors)
    }
    for k in lookup.keys():
        if k[3] == 0:
            lookup[k] = 0

    if args.visualize_cmap:
        plt.figure(figsize=(5, 1))
        plt.imshow(sorted_colors.reshape(1, -1, 4), aspect="auto")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

    dpm_2d = dpm.reshape(num_channels, -1).T
    dpm_index = np.zeros((height * width), dtype=np.float32)
    for key, value in tqdm(list(lookup.items())):
        key = np.array(key)
        dpm_index[np.all(dpm_2d == key, axis=1)] = value
    dpm_index = dpm_index.reshape(height, width)

    print("Writing output file...")
    profile["dtype"] = "float32"
    profile["count"] = 1
    profile["tiled"] = True
    profile["blockxsize"] = 512
    profile["blockysize"] = 512
    profile["interleave"] = "pixel"
    profile["compress"] = "lzw"
    profile["predictor"] = 2

    with rasterio.open(args.output_fn, "w", **profile) as f:
        f.write(dpm_index, 1)


if __name__ == "__main__":
    main()
