# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for download building footprints from Microsoft, OSM, and Google."""

import argparse
import os
import time

import dask_geopandas
import deltalake
import fiona
import fiona.transform
import geopandas as gpd
import mercantile
import osmnx as ox
import planetary_computer
import pyarrow.parquet as pq
import pystac_client
import rasterio
import s3fs
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
        "--source",
        required=True,
        type=str,
        choices=["microsoft", "osm", "google"],
        help="Options: microsoft, osm, google",
    )

    parser.add_argument(
        "--input_fn",
        type=str,
        help="Input geofile name for aoi. Options: .tif, .shp, .geojson, .gpkg",
    )

    parser.add_argument(
        "--output_dir", required=True, type=str, help="Location to building footprints"
    )

    parser.add_argument(
        "--country_alpha2_iso_code",
        required=True,
        type=str,
        help="Country Alpha2 ISO code (https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes)",
    )
    parser.add_argument(
        "--country_name",
        required=False,
        type=str,
        help="Country name to use for Microsoft footprints download (optional, overrides the ISO2 code if set))",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    return parser


def get_coordinates(input_fn):
    """Get coordinates from a GEOJSON or GEOTIFF from AOI.

    Args:
        input_fn (str): Input filename to extract geographic coordinates

    Returns:
        geom (geopandas dataframe): Shapely geom for aoi
    """
    if input_fn.endswith("tif"):
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


def get_microsoft_building_footprints(geom):
    """Searches Planetary Computer catalogue and returns Microsoft footprints for the AOI.

    Args:
        geom (polygon): Shapely geom for aoi

    Returns:
         buildings (geopandas dataframe): Set of polygons found for the aoi
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    quadkeys = [
        int(mercantile.quadkey(tile))
        for tile in mercantile.tiles(*geom.bounds, zooms=9)
    ]

    collection = catalog.get_collection("ms-buildings")
    asset = collection.assets["delta"]

    storage_options = {
        "account_name": asset.extra_fields["table:storage_options"]["account_name"],
        "sas_token": asset.extra_fields["table:storage_options"]["credential"],
    }
    table = deltalake.DeltaTable(asset.href, storage_options=storage_options)

    file_uris = table.file_uris([("quadkey", "in", quadkeys)])

    all_buildings = dask_geopandas.read_parquet(
        file_uris, storage_options=storage_options
    ).compute()

    # select valid polygons (sometimes sthe polygons are invalid)
    all_buildings["valid"] = all_buildings.geometry.apply(lambda x: x.is_valid)

    buildings = all_buildings[all_buildings["valid"]].clip(geom)
    if buildings.empty:
        return None

    buildings.drop(columns=["RegionName", "quadkey"], inplace=True)
    return buildings


def get_google_building_footprints(geom, country_iso):
    """Searches the Source Coop s3 buckets and returns Google footprints for the AOI.

    Source: https://beta.source.coop/repositories/cholmes/google-open-buildings/description

    Args:
        geom (polygon): Shapely geom for aoi
        country_iso (str): Country ISO string

    Returns:
        buildings (geopandas dataframe): Set of polygons found for the aoi
    """
    s3_path = f"s3://us-west-2.opendata.source.coop/google-research-open-buildings/v3/geoparquet-by-country/country_iso={country_iso}/{country_iso}.parquet"
    s3 = s3fs.S3FileSystem(anon=True)
    try:
        all_buildings = pq.ParquetDataset(s3_path, filesystem=s3).read().to_pandas()
    except Exception as err:
        print(f"No footprints for {err}")
        return None
    all_buildings.geometry = all_buildings.geometry.apply(
        lambda x: shapely.wkb.loads(x)
    )
    all_buildings = gpd.GeoDataFrame(all_buildings, geometry=all_buildings.geometry)
    buildings = all_buildings.clip(geom)
    return buildings


def get_osm_building_footprints(geom):
    """Searches OSM database and returns OSM building footprints for the AOI.

    Args:
        geom (polygon): Shapely geom for aoi

    Returns:
        buildings (geopandas dataframe): Set of polygons found for the aoi
    """
    tags = {"building": True}
    try:
        buildings = ox.features_from_polygon(geom.envelope, tags)
        buildings = buildings.clip(geom)
        return buildings[["building", "name", "geometry"]]
    except Exception as err:
        print(f"No footprints for {err}")
        return None


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
    start_time = time.time()
    # Handle inputs
    output_fn = os.path.join(
        args.output_dir,
        f"{args.country_alpha2_iso_code}_{args.source}_buildings_footprints.gpkg",
    )
    if os.path.exists(output_fn) and not args.overwrite:
        print(
            f"Output file '{output_fn}' already exists, specify `--overwrite` to overwrite it."
        )
        return
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Get AOI from input file
    geom = get_coordinates(args.input_fn)

    # Get footprints
    if args.source == "microsoft":
        footprints = get_microsoft_building_footprints(geom)
    elif args.source == "osm":
        footprints = get_osm_building_footprints(geom)
    elif args.source == "google":
        footprints = get_google_building_footprints(geom, args.country_alpha2_iso_code)

    if footprints is None:
        print(f"No {args.source} building footprints found for the AOI")
        return

    # Save footprints
    print(footprints.shape)
    footprints = footprints[
        footprints.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ]
    footprints.set_crs(epsg=4326, inplace=True)
    footprints.to_file(output_fn, driver="GPKG")

    print(
        f"{footprints.shape[0]} {args.source} building footprints found\
        and saved in {args.output_dir}"
    )
    end_time = time.time()
    print(f"Bldg footprint download took {end_time - start_time} seconds")


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
