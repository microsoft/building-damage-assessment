# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for download building footprints from Overture Maps."""

import argparse
import os

from bda.footprints import geodataframe

import fiona
import fiona.transform
import rasterio
import shapely


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input_fn",
        type=str,
        help="Input geofile name for aoi. Options: .tif, .shp, .geojson, .gpkg",
    )

    parser.add_argument(
        "--output_fn", required=True, type=str, help="Output filename for footprints (should end with .gpkg)"
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    return parser


def get_coordinates(input_fn):
    """Get bounds coordinates from a GeoJSON or GeoTIFF.

    Args:
        input_fn (str): Input filename extract geographic coordinates

    Returns:
        geom (geopandas dataframe): Shapely geom for AOI in EPSG:4326
    """
    if input_fn.endswith(".tif"):
        print("Input filename is a GeoTIFF, using the bounds of the file as the AOI")
        with rasterio.open(input_fn) as f:
            shape = shapely.geometry.box(
                f.bounds.left, f.bounds.bottom, f.bounds.right, f.bounds.top
            )
            crs = f.crs

    else:
        with fiona.open(input_fn) as f:
            if len(f) > 1:
                print(
                    "WARNING: Input file contains more than one feature, using the first feature as the AOI. You _probably_ don't want this though."
                )
            crs = f.crs
            geom = next(iter(f))["geometry"]
            shape = shapely.geometry.shape(geom)

    geom = shapely.geometry.mapping(shape)
    warped_geom = fiona.transform.transform_geom(crs, "EPSG:4326", geom)

    return shapely.geometry.shape(warped_geom)



def save_footprints(footprints, output_dir, footprint_source, country_code):
    """Saves building footprints to a desired location.

    Args:
        footprints (geopandas dataframe): Set of polygons found for the aoi
        output_dir (str): Location to store building footprints
        footprint_source (str): Source of building footprints
        country_code (str): Country ISO string
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    save_token = os.path.join(
        output_dir, f"{country_code}_{footprint_source}_buildings_footprints.gpkg"
    )
    footprints.to_file(save_token, driver="GPKG")


def main(args):
    """Functions for building footprint extraction.

    Args:
        args (dict): Set of input arguments
    """
    assert args.output_fn.endswith(".gpkg"), "Output filename must end with .gpkg"
    output_fn = args.output_fn
    if os.path.exists(output_fn) and not args.overwrite:
        print(
            f"Output file '{output_fn}' already exists, specify `--overwrite` to overwrite it."
        )
        return
    if os.path.exists(output_fn) and args.overwrite:
        print(f"Overwriting existing file '{output_fn}'")
        os.remove(output_fn)


    # Get AOI from input file
    shape = get_coordinates(args.input_fn)

    # Get footprints
    footprints = geodataframe("building", shape.bounds)

    # Save footprints
    footprints = footprints[["id", "geometry", "subtype", "class"]]
    footprints = footprints[
        footprints.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ]
    footprints.set_crs(epsg=4326, inplace=True)
    footprints.to_file(output_fn, driver="GPKG")

    print(
        f"{footprints.shape[0]} building footprints found and saved to {output_fn}"
    )

if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
