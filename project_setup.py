# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Preprocess the project up to the labeling tool step."""

import argparse
import json
import numpy as np
import os
import requests
import rasterio
import rasterio.warp
import shapely.geometry

from typing import Any
import azure.storage.blob
from bda.config import get_args


def add_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds the arguments for the project_setup.py script to the base parser."""
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output dataset if it already exists",
    )
    parser.add_argument(
        "--skip_health_check",
        action="store_true",
        help="Whether to skip the health check of the TiTiler endpoint",
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        required=False,
        help="The size of the grid in meters. If None, then the entire AOI is used.",
    )
    return parser


def generate_task_file(
    output_fn: str,
    geom: dict[str, Any],
    xyz_url: str,
    experiment_name: str,
) -> None:
    """Generates a task file and saves it in the save folder path.

    Args:
        output_fn: The path to save the task file.
        geom: The geometry of the task in EPSG:4326.
        xyz_url: The URL of the XYZ endpoint that serves the imagery.
        experiment_name: The name of the experiment.
    """

    output_dir = os.path.dirname(output_fn)
    os.makedirs(output_dir, exist_ok=True)
    shape = shapely.geometry.shape(geom)

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "project_name": "",
                    "name": experiment_name,
                    "instructions": """Instructions:
                                Use the polygon tool to draw a polygon around each building.
                                Use the Background class for areas that are not buildings.
                                Use the Building class for buildings that are not damaged.
                                Use the Damaged Building class for buildings that are damaged.
                                """,
                    "instructions_on_load": True,
                    "allow_wizard": True,
                    "customDataService": "",
                    "customDataServiceLabel": "Add custom",
                    "drawing_type": "polygons",
                    "layers": {
                        "Imagery": {
                            "enabled": True,
                            "type": "TileLayer",
                            "tileSize": 256,
                            "tileUrl": xyz_url,
                            "bounds": [-180, -85.5, 180, 85.5],
                        }
                    },
                    "primary_classes": {
                        "display_name": "Primary class",
                        "property_name": "class",
                        "names": [
                            "Background",
                            "Building",
                            "Damaged Building",
                        ],
                        "colors": ["#0071FE", "#00B050", "#C00000"],
                    },
                    "secondary_classes": {"property_name": "", "names": []},
                },
                "id": experiment_name,
                "bbox": list(shape.bounds),
            }
        ],
    }

    with open(output_fn, "w") as f:
        json.dump(feature_collection, f, indent=4)


def main():
    """Main function for the project_setup.py script."""
    args = get_args(description=__doc__, add_extra_parser=add_setup_parser)

    experiment_name = args["experiment_name"]
    experiment_dir = args["experiment_dir"]
    input_image_fn = args["imagery"]["raw_fn"]
    storage_account = args["infrastructure"]["storage_account"]
    assert storage_account.startswith("https://")
    if not storage_account.endswith("/"):
        storage_account += "/"
    container_name = args["infrastructure"]["container_name"]
    sas_token = args["infrastructure"]["sas_token"]
    if sas_token.startswith("?"):
        sas_token = sas_token[1:]
    relative_path = args["infrastructure"]["relative_path"]
    if not relative_path.endswith("/"):
        relative_path += "/"
    output_fn = os.path.join(experiment_dir, f"{experiment_name}_task.json")

    assert os.path.exists(input_image_fn)

    titiler_endpoint = args["infrastructure"]["titiler_endpoint"]
    assert titiler_endpoint.startswith("https://")
    if not titiler_endpoint.endswith("/"):
        titiler_endpoint += "/"

    os.makedirs(experiment_dir, exist_ok=True)

    ####################
    # check to make sure local imagery is, in fact, a COG
    with rasterio.open(input_image_fn) as f:
        overviews_exist = all([len(f.overviews(i)) > 0 for i in f.indexes])
        keys_exist = all(
            [
                "tiled" in f.profile,
                "blockxsize" in f.profile,
                "blockysize" in f.profile,
                "interleave" in f.profile,
            ]
        )
        keys_are_correct = all(
            [
                f.profile["tiled"],
                f.profile["blockxsize"] > 2,
                f.profile["blockysize"] > 2,
                f.profile["interleave"] == "pixel",
            ]
        )

        if not all([overviews_exist, keys_exist, keys_are_correct]):
            print(
                "ERROR: Imagery is not a COG, exiting (see the documentation for how to convert the imagery to COG format)."
            )
            print(f"Overviews exist: {overviews_exist}")
            print(f"Keys exist: {keys_exist}")
            print(f"Keys are correct: {keys_are_correct}")

    ####################
    # check titiler endpoint health
    if args["skip_health_check"]:
        print("Skipping TiTiler health check")
    else:
        print("Checking TiTiler health")
        response = requests.get(titiler_endpoint + "healthz")
        response.raise_for_status()
        response = response.json()
        assert response["ping"] == "pong!"

    ####################
    # copy imagery to blob
    container_client = azure.storage.blob.ContainerClient(
        storage_account,
        container_name=container_name,
        credential=sas_token,
    )
    blob_fn = os.path.join(relative_path, os.path.basename(input_image_fn))
    exists = len(list(container_client.list_blobs(name_starts_with=blob_fn))) > 0
    if exists:
        print("Imagery already exists in blob storage")
    else:
        print("Uploading imagery to blob storage")
        with open(input_image_fn, "rb") as f:
            container_client.upload_blob(
                name=blob_fn,
                data=f,
                overwrite=True,
            )
    imagery_url = f"{storage_account}{container_name}/{relative_path}{os.path.basename(input_image_fn)}?{sas_token}"
    urlencoded_imagery_url = requests.utils.quote(imagery_url, safe="")
    xyz_url = f"{titiler_endpoint}cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}?scale=1&url={urlencoded_imagery_url}"

    ####################
    # get bounds of imagery and convert to EPSG:4326
    print("Generating task file")
    with rasterio.open(input_image_fn) as f:
        bounds = shapely.geometry.box(*f.bounds)
        geom = shapely.geometry.mapping(bounds)
        if f.crs.to_epsg() != 4326:
            geom = rasterio.warp.transform_geom(f.crs, "EPSG:4326", geom)

    ####################
    # output task file
    output_fns = []
    if "grid_size" not in args:
        generate_task_file(
            output_fn=output_fn,
            geom=geom,
            xyz_url=xyz_url,
            experiment_name=experiment_name,
        )
        output_fns.append(output_fn)
    else:
        grid_size = args["grid_size"]
        geom_epsg3857 = rasterio.warp.transform_geom("EPSG:4326", "EPSG:3857", geom)
        shape = shapely.geometry.shape(geom_epsg3857)
        minx, miny, maxx, maxy = shape.bounds

        i = 0
        for x in np.arange(minx, maxx, grid_size):
            for y in np.arange(miny, maxy, grid_size):
                grid = shapely.geometry.box(x, y, x + grid_size, y + grid_size)
                grid = shapely.geometry.mapping(grid)
                if shape.intersects(shapely.geometry.shape(grid)):
                    t_output_fn = output_fn.replace("_task.json", f"_task_{i}.json")
                    t_geom = rasterio.warp.transform_geom("EPSG:3857", "EPSG:4326", grid)
                    generate_task_file(
                        output_fn=t_output_fn,
                        geom=t_geom,
                        xyz_url=xyz_url,
                        experiment_name=experiment_name
                    )
                    output_fns.append(t_output_fn)
                    i += 1

    for output_fn in output_fns:
        with open(output_fn, "rb") as f:
            container_client.upload_blob(
                name=os.path.join(
                    relative_path,
                    os.path.basename(output_fn),
                ),
                data=f,
                overwrite=True,
            )
        task_url = f"{storage_account}{container_name}/{relative_path}{os.path.basename(output_fn)}"
        urlencoded_task_url = requests.utils.quote(task_url, safe="")
        labeling_tool_url = f"https://microsoft.github.io/satellite-imagery-labeling-tool/src/labeler.html?taskUrl={urlencoded_task_url}?{sas_token}"
        print(f"Labeling tool URL: {labeling_tool_url}")


if __name__ == "__main__":
    main()
