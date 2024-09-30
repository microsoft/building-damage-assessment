# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for merging building footprints with damage predictions."""

import argparse
import os

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.mask
import shapely.geometry
from tqdm import tqdm


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--footprints_fn", type=str, required=True, help="Path to the footprint file"
    )
    parser.add_argument(
        "--predictions_fn", type=str, required=True, help="Path to the prediction file"
    )
    parser.add_argument(
        "--output_fn", type=str, required=True, help="Path to the output file"
    )

    return parser


def main(args):
    """Main function for the merge_with_building_footprints.py script."""
    with rasterio.open(args.predictions_fn, "r") as src:
        predictions_crs = src.crs.to_string()

    with fiona.open(args.footprints_fn, "r") as src:
        footprints_crs = src.crs.to_string()

    # Clip building footprints to image data mask
    projected_building_geoms = []
    with fiona.open(args.footprints_fn) as f:
        for row in tqdm(f):
            projected_geom = fiona.transform.transform_geom(
                footprints_crs, predictions_crs, row["geometry"]
            )
            projected_building_geoms.append(projected_geom)

    ############################################
    # Read predictions within building footprints
    # and track damage values
    ############################################
    damage_vals_per_geom = []
    unknown_val_per_geom = []
    print(f"Reading predictions from {args.predictions_fn}")
    with rasterio.open(args.predictions_fn) as f:
        # Compute the fraction of damage per geometry and buffer size option
        for building_geom in tqdm(projected_building_geoms):
            t_dmg_vals = []
            for buffer in [0, 10, 20]:
                building_shape = shapely.geometry.shape(building_geom).buffer(buffer)

                building_mask, transform = rasterio.mask.mask(
                    f, [building_shape], crop=True, nodata=0, filled=True
                )
                vals, counts = np.unique(building_mask, return_counts=True)
                val_counts = dict(zip(vals, counts))

                N = 0
                for k, v in val_counts.items():
                    if k != 0:
                        N += v

                if 3 in val_counts:
                    fraction_damaged = min(val_counts[3] / N, 1)
                else:
                    fraction_damaged = 0
                t_dmg_vals.append(fraction_damaged)
            damage_vals_per_geom.append(t_dmg_vals)

        # Compute the fraction of unknown (cloud covered) pixels per geometry
        for building_geom in tqdm(projected_building_geoms):
            building_shape = shapely.geometry.shape(building_geom)

            building_mask, transform = rasterio.mask.mask(
                f, [building_shape], crop=True, nodata=0, filled=True
            )
            vals, counts = np.unique(building_mask, return_counts=True)
            val_counts = dict(zip(vals, counts))

            N = 0
            for k, v in val_counts.items():
                if k != 0:
                    N += v

            if 4 in val_counts:
                fraction_unknown = val_counts[4] / N
            else:
                fraction_unknown = 0
            unknown_val_per_geom.append(fraction_unknown)

    ############################################
    # Write damage values to file
    ############################################
    schema = {
        "geometry": "MultiPolygon",
        "properties": {
            "id": "int",
            "damage_pct_0m": "float",
            "damage_pct_10m": "float",
            "damage_pct_20m": "float",
            "damaged": "int",
            "unknown_pct": "float",
        },
    }

    if os.path.exists(args.output_fn):
        os.remove(args.output_fn)

    with fiona.open(
        args.output_fn, "w", driver="GPKG", crs=predictions_crs, schema=schema
    ) as f:
        for i, geom in enumerate(tqdm(projected_building_geoms)):
            shape = shapely.geometry.shape(geom)
            if geom["type"] == "Polygon":
                geom = shapely.geometry.mapping(shapely.geometry.MultiPolygon([shape]))

            row = {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "id": i,
                    "damage_pct_0m": damage_vals_per_geom[i][0],
                    "damage_pct_10m": damage_vals_per_geom[i][1],
                    "damage_pct_20m": damage_vals_per_geom[i][2],
                    "damaged": 1 if damage_vals_per_geom[i][0] > 0 else 0,
                    "unknown_pct": unknown_val_per_geom[i],
                },
            }
            f.write(row)

    print(f"Output written to {args.output_fn}")
    damage_vals_per_geom = np.array(damage_vals_per_geom)
    breakpoints = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    for i in range(1, len(breakpoints)):
        count = np.sum(
            (damage_vals_per_geom[:, 0] >= breakpoints[i - 1])
            & (damage_vals_per_geom[:, 0] < breakpoints[i])
        )
        print(
            f"- {count} buildings with damage fraction between {breakpoints[i-1]*100:0.0f}% and {breakpoints[i]*100:0.0f}%"
        )


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
