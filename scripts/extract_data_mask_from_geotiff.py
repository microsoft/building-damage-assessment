# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for creating a valid data mask (GeoJSON) from an input GeoTIFF.

Often we want to create a GeoJSON that contains a shape that outlines the valid data in
a satellite imagery scene -- e.g., to subset a building footprint dataset by the
area covered by satellite imagery.

NOTE: The output GeoJSON will be saved in EPSG:4326, regardless of the input CRS.
"""

import argparse
import os
import sys

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.features
import shapely.geometry


def main():
    """Main function for the extract_data_mask_from_geotiff.py script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_fn", type=str, required=True, help="Path to input GeoTIFF file"
    )
    parser.add_argument(
        "--output_fn", type=str, required=True, help="Path to output GeoJSON"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    args = parser.parse_args()

    # Need to check this as we assume the fiona driver is GeoJSON later
    assert args.output_fn.endswith(".geojson")

    if not os.path.exists(args.input_fn):
        print(f"ERROR: Input file does not exist: {args.input_fn}")
        sys.exit(1)

    if os.path.exists(args.output_fn) and not args.overwrite:
        print(f"ERROR: Output file already exists: {args.output_fn}, use `--overwrite`")
        sys.exit(1)
    elif os.path.exists(args.output_fn) and args.overwrite:
        print(f"WARNING: Overwriting output file: {args.output_fn}")
        os.remove(args.output_fn)

    output_dir = os.path.dirname(args.output_fn)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Read input
    with rasterio.open(args.input_fn) as f:
        data = f.read()
        transform = f.transform
        crs = f.crs.to_string()

    print("Loaded data with shape", data.shape)
    num_channels, _, __ = data.shape
    mask = ((data == 0).sum(axis=0) != num_channels).astype(np.uint8)

    features = list(rasterio.features.shapes(mask, transform=transform))
    geoms = []
    for geom, val in features:
        if val == 1:
            geoms.append(geom)

    if len(geoms) > 1:
        print(
            f"WARNING: Found {len(geoms)} valid data regions, this might be unexpected."
        )

    # Write output
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}

    with fiona.open(
        args.output_fn, "w", driver="GeoJSON", crs="EPSG:4326", schema=schema
    ) as f:
        for i, geom in enumerate(geoms):
            shape = shapely.geometry.shape(geom)
            if crs == "EPSG:4326":  # Simplify by approximately 10 meters
                geom = shapely.geometry.mapping(shape.simplify(10 / 111_139))
            else:  # We assume that anything that is not EPSG:4326 is projected
                geom = shapely.geometry.mapping(shape.simplify(10))
                geom = fiona.transform.transform_geom(crs, "EPSG:4326", geom)

            row = {"type": "Feature", "geometry": geom, "properties": {"id": i}}
            f.write(row)


if __name__ == "__main__":
    main()
