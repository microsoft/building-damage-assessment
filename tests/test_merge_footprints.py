# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the merge_with_building_footprints.py script."""

import os
import subprocess
import sys
import numpy as np
import pytest
import fiona
import rasterio
import shapely.geometry

def create_predictions_raster(filename):
    crs = "EPSG:4326"
    transform = rasterio.transform.from_origin(0, 10, 1, 1)
    # 3 = damage, 2 = built, 4 = unknown, 0 = nodata
    data = np.zeros((1, 10, 10), dtype=np.uint8)
    data[0, 1:4, 1:4] = 3  # damage region (9 pixels)
    with rasterio.open(
        filename, "w", driver="GTiff", height=10, width=10, count=1, dtype=np.uint8, crs=crs, transform=transform
    ) as dst:
        dst.write(data)

def create_footprints(filename, count):
    schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "str",
        }
    }
    crs = "EPSG:4326"
    with fiona.open(filename, "w", driver="GPKG", crs=crs, schema=schema) as dst:
        for i in range(count):
            offset = i * 2
            geom = shapely.geometry.box(1 + offset, 1 + offset, 3 + offset, 3 + offset)
            dst.write({
                "geometry": shapely.geometry.mapping(geom),
                "properties": {"id": f"building_{i}"}
            })

def run_merge_script(footprints_fn, predictions_fn, output_fn):
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "merge_with_building_footprints.py"
    )
    cmd = [
        sys.executable,
        script_path,
        "--footprints_fn", footprints_fn,
        "--predictions_fn", predictions_fn,
        "--output_fn", output_fn
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_empty_footprints(tmp_path):
    """Test empty footprints scenario yields a valid empty output with exit code 0."""
    predictions_fn = str(tmp_path / "predictions.tif")
    footprints_fn = str(tmp_path / "empty_footprints.gpkg")
    output_fn = str(tmp_path / "output.gpkg")

    create_predictions_raster(predictions_fn)
    create_footprints(footprints_fn, 0)

    res = run_merge_script(footprints_fn, predictions_fn, output_fn)
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"

    assert os.path.exists(output_fn)
    with fiona.open(output_fn) as src:
        assert len(src) == 0
        assert src.crs.to_string() == "EPSG:4326"
        assert "damage_pct_0m" in src.schema["properties"]


def test_one_footprint(tmp_path):
    """Test single footprint processes correctly and prints statistics."""
    predictions_fn = str(tmp_path / "predictions.tif")
    footprints_fn = str(tmp_path / "one_footprint.gpkg")
    output_fn = str(tmp_path / "output.gpkg")

    create_predictions_raster(predictions_fn)
    create_footprints(footprints_fn, 1)

    res = run_merge_script(footprints_fn, predictions_fn, output_fn)
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"

    assert "1 buildings with damage fraction" in res.stdout
    assert os.path.exists(output_fn)
    with fiona.open(output_fn) as src:
        assert len(src) == 1


def test_multiple_footprints(tmp_path):
    """Test multiple footprints process correctly and print statistics."""
    predictions_fn = str(tmp_path / "predictions.tif")
    footprints_fn = str(tmp_path / "multiple_footprints.gpkg")
    output_fn = str(tmp_path / "output.gpkg")

    create_predictions_raster(predictions_fn)
    create_footprints(footprints_fn, 3)

    res = run_merge_script(footprints_fn, predictions_fn, output_fn)
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"

    assert "3 buildings with damage fraction" in res.stdout
    assert os.path.exists(output_fn)
    with fiona.open(output_fn) as src:
        assert len(src) == 3
