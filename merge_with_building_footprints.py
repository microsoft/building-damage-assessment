# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for merging building footprints with damage predictions."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "Number of parallel worker processes. Defaults to all available CPUs; "
            "pass 1 to run serially."
        ),
    )

    return parser


def compute_damage_and_built(src, building_geom):
    """Compute damage and built fractions at 0m/10m/20m buffers for one geometry.

    Args:
        src: An open rasterio dataset of the predictions.
        building_geom: A GeoJSON-like geometry in the predictions CRS.

    Returns:
        Tuple of (damage_fractions, built_fractions), each a list ordered by the
        [0, 10, 20] meter buffers.
    """
    t_dmg_vals = []
    t_built_vals = []
    for buffer in [0, 10, 20]:
        building_shape = shapely.geometry.shape(building_geom).buffer(buffer)

        try:
            building_mask, _ = rasterio.mask.mask(
                src, [building_shape], crop=True, nodata=0, filled=True
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

            if 2 in val_counts:
                fraction_built = min(val_counts[2] / N, 1)
            else:
                fraction_built = 0
            t_built_vals.append(fraction_built)
            t_dmg_vals.append(fraction_damaged)
        except ValueError:
            t_built_vals.append(0)
            t_dmg_vals.append(0)

    return t_dmg_vals, t_built_vals


def compute_unknown(src, building_geom):
    """Compute the fraction of unknown (cloud covered) pixels for one geometry.

    Args:
        src: An open rasterio dataset of the predictions.
        building_geom: A GeoJSON-like geometry in the predictions CRS.

    Returns:
        The fraction of unknown (class 4) pixels within the geometry.
    """
    building_shape = shapely.geometry.shape(building_geom)

    try:
        building_mask, _ = rasterio.mask.mask(
            src, [building_shape], crop=True, nodata=0, filled=True
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
    except ValueError:
        fraction_unknown = 1

    return fraction_unknown


def process_damage_values(geom_idx_tuple, predictions_fn):
    """Worker wrapper that opens the predictions file and computes damage values.

    Each worker process opens the raster independently because open file handles
    cannot be shared across processes.

    Args:
        geom_idx_tuple: Tuple of (index, geometry).
        predictions_fn: Path to the predictions file.

    Returns:
        Tuple of (index, damage_fractions, built_fractions).
    """
    idx, building_geom = geom_idx_tuple
    with rasterio.open(predictions_fn) as src:
        t_dmg_vals, t_built_vals = compute_damage_and_built(src, building_geom)
    return idx, t_dmg_vals, t_built_vals


def process_unknown_values(geom_idx_tuple, predictions_fn):
    """Worker wrapper that opens the predictions file and computes unknown values.

    Args:
        geom_idx_tuple: Tuple of (index, geometry).
        predictions_fn: Path to the predictions file.

    Returns:
        Tuple of (index, unknown_fraction).
    """
    idx, building_geom = geom_idx_tuple
    with rasterio.open(predictions_fn) as src:
        fraction_unknown = compute_unknown(src, building_geom)
    return idx, fraction_unknown


def main(args):
    """Main function for the merge_with_building_footprints.py script."""
    with rasterio.open(args.predictions_fn, "r") as src:
        predictions_crs = src.crs.to_string()

    with fiona.open(args.footprints_fn, "r") as src:
        footprints_crs = src.crs.to_string()

    projected_building_geoms = []
    projected_building_ids = []
    with fiona.open(args.footprints_fn) as f:
        for row in tqdm(f, desc="Projecting building footprints"):
            projected_geom = fiona.transform.transform_geom(
                footprints_crs, predictions_crs, row["geometry"]
            )
            projected_building_geoms.append(projected_geom)
            projected_building_ids.append(row["properties"]["id"])  # these are Overture IDs, we want to keep these so we can re-id buildings

    ############################################
    # Read predictions within building footprints
    # and track damage values
    ############################################
    num_geoms = len(projected_building_geoms)
    damage_vals_per_geom = [None] * num_geoms
    built_vals_per_geom = [None] * num_geoms
    unknown_val_per_geom = [None] * num_geoms

    geom_indices = list(enumerate(projected_building_geoms))
    print(f"Reading predictions from {args.predictions_fn}")

    if args.num_workers == 1:
        # Serial fast-path: open the raster once and reuse the handle for all buildings
        with rasterio.open(args.predictions_fn) as f:
            for idx, building_geom in tqdm(
                geom_indices, desc="Computing damage fractions"
            ):
                t_dmg_vals, t_built_vals = compute_damage_and_built(f, building_geom)
                damage_vals_per_geom[idx] = t_dmg_vals
                built_vals_per_geom[idx] = t_built_vals

            for idx, building_geom in tqdm(
                geom_indices, desc="Computing unknown fractions"
            ):
                unknown_val_per_geom[idx] = compute_unknown(f, building_geom)
    else:
        # Parallel path: distribute per-building work across processes
        process_damage_fn = partial(
            process_damage_values, predictions_fn=args.predictions_fn
        )
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_damage_fn, geom_tuple): geom_tuple
                for geom_tuple in geom_indices
            }
            with tqdm(total=len(futures), desc="Computing damage fractions") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, dmg_vals, built_vals = future.result()
                        damage_vals_per_geom[idx] = dmg_vals
                        built_vals_per_geom[idx] = built_vals
                    except Exception as e:
                        idx = futures[future][0]
                        print(f"Error processing geometry {idx}: {e}")
                        damage_vals_per_geom[idx] = [0, 0, 0]
                        built_vals_per_geom[idx] = [0, 0, 0]
                    pbar.update(1)

        process_unknown_fn = partial(
            process_unknown_values, predictions_fn=args.predictions_fn
        )
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_unknown_fn, geom_tuple): geom_tuple
                for geom_tuple in geom_indices
            }
            with tqdm(total=len(futures), desc="Computing unknown fractions") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, unknown_val = future.result()
                        unknown_val_per_geom[idx] = unknown_val
                    except Exception as e:
                        idx = futures[future][0]
                        print(f"Error processing geometry {idx}: {e}")
                        unknown_val_per_geom[idx] = 1
                    pbar.update(1)

    ############################################
    # Write damage values to file
    ############################################
    schema = {
        "geometry": "MultiPolygon",
        "properties": {
            "id": "str",
            "damage_pct_0m": "float",
            "damage_pct_10m": "float",
            "damage_pct_20m": "float",
            "built_pct_0m": "float",
            "damaged": "int",
            "unknown_pct": "float",
        },
    }

    if os.path.exists(args.output_fn):
        os.remove(args.output_fn)

    # Accumulate rows before writing as writerecords is much faster than writing one at a time
    rows = []
    for i, geom in enumerate(tqdm(projected_building_geoms, desc="Preparing output rows")):
        shape = shapely.geometry.shape(geom)
        if geom["type"] == "Polygon":
            geom = shapely.geometry.mapping(shapely.geometry.MultiPolygon([shape]))

        row = {
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "id": projected_building_ids[i],
                "damage_pct_0m": damage_vals_per_geom[i][0],
                "damage_pct_10m": damage_vals_per_geom[i][1],
                "damage_pct_20m": damage_vals_per_geom[i][2],
                "built_pct_0m": built_vals_per_geom[i][0],
                "damaged": 1 if damage_vals_per_geom[i][0] > 0 else 0,
                "unknown_pct": unknown_val_per_geom[i],
            },
        }
        rows.append(row)

    print("Writing output file...")
    with fiona.open(
        args.output_fn, "w", driver="GPKG", crs=predictions_crs, schema=schema
    ) as f:
        f.writerecords(rows)

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
