# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for dpm_intersection.py."""

import os
import subprocess
import sys
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import fiona
import fiona.transform
import shapely.geometry


@pytest.fixture
def temp_dpm_and_footprints(tmp_path):
    """Creates a temporary DPM GeoTIFF and GPKG building footprint file in EPSG:4326."""
    dpm_fn = str(tmp_path / "dpm.tif")
    buildings_fn = str(tmp_path / "buildings.gpkg")

    # 10x10 raster in EPSG:4326 from (x=0, y=10) down to (x=10, y=0), pixel size 1x1
    transform = from_origin(0, 10, 1, 1)
    # All pixels have DPM value 0.85
    data = np.ones((1, 10, 10), dtype=np.float32) * 0.85

    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(dpm_fn, "w", **profile) as ds:
        ds.write(data)

    # Building 1: triangle from (1,1) to (5,1) to (1,5) -> occupies part of bounding box
    # Building 2: polygon completely outside raster bounds (e.g. x=50..55, y=50..55)
    poly1 = shapely.geometry.Polygon([(1, 1), (5, 1), (1, 5), (1, 1)])
    poly2 = shapely.geometry.Polygon([(50, 50), (55, 50), (55, 55), (50, 50)])

    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(buildings_fn, "w", driver="GPKG", crs="EPSG:4326", schema=schema) as ds:
        ds.writerecords(
            [
                {"geometry": shapely.geometry.mapping(poly1), "properties": {"id": 1}},
                {"geometry": shapely.geometry.mapping(poly2), "properties": {"id": 2}},
            ]
        )

    return dpm_fn, buildings_fn


def test_dpm_intersection_accurate_masking(tmp_path, temp_dpm_and_footprints):
    """Test that average_dpm only averages pixels inside the building geometry."""
    dpm_fn, buildings_fn = temp_dpm_and_footprints
    output_fn = str(tmp_path / "output.gpkg")

    cmd = [
        sys.executable,
        "dpm_intersection.py",
        "--input_dpm_fn",
        dpm_fn,
        "--input_buildings_fn",
        buildings_fn,
        "--output_fn",
        output_fn,
        "--threshold",
        "0.5",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"

    with fiona.open(output_fn) as ds:
        features = list(ds)
        assert len(features) == 2

        # Building 1: inside raster, pixels are 0.85 -> average_dpm should be ~0.85
        # In the buggy version, np.mean(mask) would include the 0 padding and be ~0.53
        b1_props = features[0]["properties"]
        assert np.isclose(b1_props["average_dpm"], 0.85, atol=1e-4)
        assert b1_props["damaged"] == 1

        # Building 2: out of raster bounds -> average_dpm should be 0.0, damaged=0
        b2_props = features[1]["properties"]
        assert b2_props["average_dpm"] == 0.0
        assert b2_props["damaged"] == 0


def test_dpm_intersection_nan_nodata(tmp_path):
    """Test that DPM rasters with NaN nodata don't cause NaN average_dpm."""
    dpm_fn = str(tmp_path / "dpm_nan.tif")
    buildings_fn = str(tmp_path / "buildings_nan.gpkg")
    output_fn = str(tmp_path / "output_nan.gpkg")

    transform = from_origin(0, 10, 1, 1)
    data = np.ones((1, 10, 10), dtype=np.float32) * 0.7
    # Set one pixel outside the building to NaN
    data[0, 0, 0] = np.nan

    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.open(dpm_fn, "w", **profile) as ds:
        ds.write(data)

    poly = shapely.geometry.Polygon([(1, 1), (4, 1), (1, 4), (1, 1)])
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(buildings_fn, "w", driver="GPKG", crs="EPSG:4326", schema=schema) as ds:
        ds.writerecords([{"geometry": shapely.geometry.mapping(poly), "properties": {"id": 1}}])

    cmd = [
        sys.executable,
        "dpm_intersection.py",
        "--input_dpm_fn",
        dpm_fn,
        "--input_buildings_fn",
        buildings_fn,
        "--output_fn",
        output_fn,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"

    with fiona.open(output_fn) as ds:
        feat = next(iter(ds))
        avg_dpm = feat["properties"]["average_dpm"]
        assert not np.isnan(avg_dpm)
        assert np.isclose(avg_dpm, 0.7, atol=1e-4)


def test_dpm_intersection_crs_transformation(tmp_path):
    """Test intersecting building footprints in EPSG:4326 with a DPM in UTM (EPSG:32631)."""
    dpm_fn = str(tmp_path / "dpm_utm.tif")
    buildings_fn = str(tmp_path / "buildings_4326.gpkg")
    output_fn = str(tmp_path / "output_transformed.gpkg")

    # UTM Zone 31N: (500000, 10000)
    transform = from_origin(500000, 10000, 10, 10)
    data = np.ones((1, 10, 10), dtype=np.float32) * 0.9

    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32631",
        "transform": transform,
    }
    with rasterio.open(dpm_fn, "w", **profile) as ds:
        ds.write(data)

    # Footprint in EPSG:4326 that maps to ~ (500050, 9950) in UTM
    poly_utm = shapely.geometry.Polygon(
        [(500020, 9920), (500080, 9920), (500080, 9980), (500020, 9980)]
    )
    geom_4326 = fiona.transform.transform_geom(
        "EPSG:32631", "EPSG:4326", shapely.geometry.mapping(poly_utm)
    )

    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(buildings_fn, "w", driver="GPKG", crs="EPSG:4326", schema=schema) as ds:
        ds.writerecords([{"geometry": geom_4326, "properties": {"id": 101}}])

    cmd = [
        sys.executable,
        "dpm_intersection.py",
        "--input_dpm_fn",
        dpm_fn,
        "--input_buildings_fn",
        buildings_fn,
        "--output_fn",
        output_fn,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"

    with fiona.open(output_fn) as ds:
        feat = next(iter(ds))
        assert np.isclose(feat["properties"]["average_dpm"], 0.9, atol=1e-4)


def test_dpm_intersection_local_output_and_overwrite(tmp_path, temp_dpm_and_footprints):
    """Test relative output filename (no parent dir) and --overwrite handling."""
    dpm_fn, buildings_fn = temp_dpm_and_footprints
    local_output_fn = "test_local_output_tmp.gpkg"

    if os.path.exists(local_output_fn):
        os.remove(local_output_fn)

    try:
        cmd = [
            sys.executable,
            "dpm_intersection.py",
            "--input_dpm_fn",
            dpm_fn,
            "--input_buildings_fn",
            buildings_fn,
            "--output_fn",
            local_output_fn,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Local path failed: {res.stderr}"
        assert os.path.exists(local_output_fn)

        # Without overwrite, should exit with code 1
        res_no_ow = subprocess.run(cmd, capture_output=True, text=True)
        assert res_no_ow.returncode == 1

        # With overwrite, should succeed
        res_ow = subprocess.run(cmd + ["--overwrite"], capture_output=True, text=True)
        assert res_ow.returncode == 0, f"Overwrite failed: {res_ow.stderr}"
    finally:
        if os.path.exists(local_output_fn):
            os.remove(local_output_fn)


def test_dpm_intersection_empty_input(tmp_path, temp_dpm_and_footprints):
    """Test empty building footprints layer does not crash."""
    dpm_fn, _ = temp_dpm_and_footprints
    empty_buildings_fn = str(tmp_path / "empty.gpkg")
    output_fn = str(tmp_path / "empty_out.gpkg")

    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(empty_buildings_fn, "w", driver="GPKG", crs="EPSG:4326", schema=schema):
        pass

    cmd = [
        sys.executable,
        "dpm_intersection.py",
        "--input_dpm_fn",
        dpm_fn,
        "--input_buildings_fn",
        empty_buildings_fn,
        "--output_fn",
        output_fn,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Empty input failed: {res.stderr}"
    assert os.path.exists(output_fn)
