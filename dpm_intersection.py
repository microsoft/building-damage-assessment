# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for intersecting damage proxy maps with a building footprint layer."""

import argparse
import os
import sys

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.errors
import rasterio.mask
from tqdm import tqdm


# TODO: integrate with config file setup
def setup_parser() -> argparse.ArgumentParser:
    """Adds the arguments for the inference.py script to the base parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dpm_fn", type=str, help="Input raw DPM file")
    parser.add_argument(
        "--input_buildings_fn", type=str, help="Input building footprint file"
    )
    parser.add_argument("--output_fn", type=str, help="Filename of output file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrites the output file if it exist",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=False,
        help="Threshold value used to compute damaged property (damaged property isn't computed if `None`)",
    )

    return parser


def main() -> None:
    """Main function for the dpm_intersection.py script."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.threshold is not None:
        assert 0 <= args.threshold < 1
    if os.path.exists(args.output_fn) and not args.overwrite:
        print(
            f"Output file '{args.output_fn}' already exists, use `--overwrite` to"
            + " overwrite."
        )
        sys.exit(1)
    if os.path.exists(args.output_fn) and args.overwrite:
        os.remove(args.output_fn)

    output_dir = os.path.dirname(args.output_fn)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Intersect the DPM with the building footprint layer
    new_features = []
    with rasterio.open(args.input_dpm_fn) as raster_f:
        raster_crs = raster_f.crs.to_string()
        with fiona.open(args.input_buildings_fn) as f:
            footprints_crs = f.crs.to_string() if f.crs else raster_crs
            schema = f.schema.copy()
            crs = footprints_crs
            for feature in tqdm(f):
                geom = feature["geometry"]
                if geom is None:
                    continue

                if footprints_crs != raster_crs:
                    geom_in_raster_crs = fiona.transform.transform_geom(
                        footprints_crs, raster_crs, geom
                    )
                else:
                    geom_in_raster_crs = geom

                avg_dpm = 0.0
                try:
                    mask, _ = rasterio.mask.mask(
                        raster_f, [geom_in_raster_crs], crop=True, all_touched=True, filled=False
                    )
                    compressed = mask.compressed()
                    valid_vals = compressed[~np.isnan(compressed)]
                    if len(valid_vals) > 0:
                        avg_dpm = float(np.mean(valid_vals))
                except (ValueError, rasterio.errors.RasterioError):
                    # Geometry does not overlap raster or has no valid pixels
                    pass

                properties = dict(feature["properties"])
                properties["average_dpm"] = avg_dpm
                new_features.append(fiona.Feature(geometry=geom, properties=properties))

    # Write the output
    schema["properties"]["average_dpm"] = "float"

    if args.threshold is not None:
        schema["properties"]["damaged"] = "int"
        t_new_features = []
        for i in range(len(new_features)):
            properties = dict(new_features[i]["properties"])
            properties["damaged"] = int(properties["average_dpm"] > args.threshold)
            t_new_features.append(
                fiona.Feature(
                    geometry=new_features[i]["geometry"], properties=properties
                )
            )
        new_features = t_new_features

    with fiona.open(args.output_fn, "w", driver="GPKG", crs=crs, schema=schema) as f:
        f.writerecords(new_features)

    # Print some statistics
    damage_vals_per_geom = []
    for feature in tqdm(new_features):
        damage_vals_per_geom.append(feature["properties"]["average_dpm"])
    damage_vals_per_geom_arr = np.array(damage_vals_per_geom)

    breakpoints: list[float] = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    for i in range(1, len(breakpoints)):
        if len(damage_vals_per_geom_arr) > 0:
            count = np.sum(
                (damage_vals_per_geom_arr >= breakpoints[i - 1])
                & (damage_vals_per_geom_arr < breakpoints[i])
            )
        else:
            count = 0

        lower_pct = f"{breakpoints[i-1]*100:0.0f}"
        upper_pct = f"{breakpoints[i]*100:0.0f}"

        print(
            f"- {count} buildings with average DPM value between {lower_pct}% and"
            + f" {upper_pct}%"
        )


if __name__ == "__main__":
    main()
