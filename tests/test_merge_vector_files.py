# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the merge_vector_files.py utility script."""

import os
import subprocess
import sys
import pytest
import fiona
import shapely.geometry

# Add workspace root to system path to import merge script if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_dummy_gpkg(filepath, feature_id, crs=None, coords=None):
    schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "str",
        }
    }
    if coords is None:
        if crs == "EPSG:32630":
            coords = [(499950, 5999950), (500050, 5999950), (500050, 6000050), (499950, 6000050), (499950, 5999950)]
        else:
            coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]

    geom = shapely.geometry.Polygon(coords)
    with fiona.open(filepath, "w", driver="GPKG", crs=crs, schema=schema) as dst:
        dst.write({
            "geometry": shapely.geometry.mapping(geom),
            "properties": {"id": str(feature_id)}
        })

def run_merge_script(args):
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "merge_vector_files.py"
    )
    cmd = [sys.executable, script_path] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def test_overwrite_option(tmp_path):
    """Test A1: --overwrite option does not crash with AttributeError and merges files."""
    input1 = str(tmp_path / "input1.gpkg")
    input2 = str(tmp_path / "input2.gpkg")
    output = str(tmp_path / "output.gpkg")

    create_dummy_gpkg(input1, "feature_1", crs="EPSG:4326")
    create_dummy_gpkg(input2, "feature_2", crs="EPSG:4326")
    create_dummy_gpkg(output, "old_feature", crs="EPSG:4326")

    res = run_merge_script([input1, input2, output, "--overwrite"])
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"

    assert os.path.exists(output)
    with fiona.open(output) as src:
        features = list(src)
        assert len(features) == 2
        ids = [f["properties"]["id"] for f in features]
        assert "feature_1" in ids
        assert "feature_2" in ids


def test_current_directory_output(tmp_path, monkeypatch):
    """Test A2: Specifying output file in current directory does not raise FileNotFoundError."""
    input1 = str(tmp_path / "input1.gpkg")
    input2 = str(tmp_path / "input2.gpkg")

    create_dummy_gpkg(input1, "feature_1", crs="EPSG:4326")
    create_dummy_gpkg(input2, "feature_2", crs="EPSG:4326")

    # Run from a temporary working directory using relative output path
    monkeypatch.chdir(tmp_path)
    output_rel = "output_rel.gpkg"

    res = run_merge_script([input1, input2, output_rel])
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"
    assert os.path.exists(str(tmp_path / output_rel))

    # Test nested directory creation is still preserved
    output_nested = "nested_dir/output_nested.gpkg"
    res = run_merge_script([input1, input2, output_nested])
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"
    assert os.path.exists(str(tmp_path / output_nested))


def test_mixed_crs_reprojection_forward(tmp_path):
    """Test A3: Mixed CRS (WGS84 first, UTM second) transforms UTM coordinates to WGS84."""
    input_wgs = str(tmp_path / "input_wgs.gpkg")
    input_utm = str(tmp_path / "input_utm.gpkg")
    output = str(tmp_path / "output_mixed_forward.gpkg")

    create_dummy_gpkg(input_wgs, "feature_wgs", crs="EPSG:4326")
    create_dummy_gpkg(input_utm, "feature_utm", crs="EPSG:32630")

    res = run_merge_script([input_wgs, input_utm, output])
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"

    with fiona.open(output) as src:
        # Expected destination CRS is the first file's CRS (WGS84)
        assert src.crs.to_string() == "EPSG:4326"
        features = list(src)
        assert len(features) == 2

        feat_wgs = next(f for f in features if f["properties"]["id"] == "feature_wgs")
        feat_utm = next(f for f in features if f["properties"]["id"] == "feature_utm")

        # Check original raw WGS84 feature remains correct
        geom_wgs = shapely.geometry.shape(feat_wgs["geometry"])
        assert geom_wgs.bounds == (0.0, 0.0, 1.0, 1.0)

        # Check UTM feature was reprojected and matches expected lat/lon coords (-3, 54.148)
        geom_utm = shapely.geometry.shape(feat_utm["geometry"])
        minx, miny, maxx, maxy = geom_utm.bounds

        # Raw UTM coords were around 500000, 6000000. Under WGS84 they must be transformed.
        assert -3.01 < minx < -2.99
        assert 54.14 < miny < 54.15
        assert -3.00 < maxx < -2.98
        assert 54.14 < maxy < 54.15


def test_mixed_crs_reprojection_reverse(tmp_path):
    """Test A3: Mixed CRS in reverse (UTM first, WGS84 second) transforms WGS84 coordinates to UTM."""
    input_utm = str(tmp_path / "input_utm.gpkg")
    input_wgs = str(tmp_path / "input_wgs.gpkg")
    output = str(tmp_path / "output_mixed_reverse.gpkg")

    create_dummy_gpkg(input_utm, "feature_utm", crs="EPSG:32630")
    create_dummy_gpkg(input_wgs, "feature_wgs", crs="EPSG:4326")

    res = run_merge_script([input_utm, input_wgs, output])
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"

    with fiona.open(output) as src:
        # Expected destination CRS is the first file's CRS (UTM EPSG:32630)
        assert src.crs.to_string() == "EPSG:32630"
        features = list(src)
        assert len(features) == 2

        feat_utm = next(f for f in features if f["properties"]["id"] == "feature_utm")
        feat_wgs = next(f for f in features if f["properties"]["id"] == "feature_wgs")

        # Check original raw UTM feature remains correct
        geom_utm = shapely.geometry.shape(feat_utm["geometry"])
        assert geom_utm.bounds == (499950.0, 5999950.0, 500050.0, 6000050.0)

        # Check WGS84 feature was reprojected and matches expected UTM coordinates
        geom_wgs = shapely.geometry.shape(feat_wgs["geometry"])
        minx, miny, maxx, maxy = geom_wgs.bounds
        assert minx > 500000  # WGS84 bounds (0,0,1,1) in UTM Zone 30N are large positive values
        assert miny >= 0.0


def test_same_crs(tmp_path):
    """Test same CRS merges correctly without unnecessary reprojections."""
    input1 = str(tmp_path / "input1.gpkg")
    input2 = str(tmp_path / "input2.gpkg")
    output = str(tmp_path / "output_same.gpkg")

    create_dummy_gpkg(input1, "feature_1", crs="EPSG:4326")
    create_dummy_gpkg(input2, "feature_2", crs="EPSG:4326")

    res = run_merge_script([input1, input2, output])
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"

    with fiona.open(output) as src:
        assert src.crs.to_string() == "EPSG:4326"
        features = list(src)
        assert len(features) == 2


def test_no_crs(tmp_path):
    """Test that a file with no CRS preserves raw coordinates without raising exceptions."""
    input_no_crs = str(tmp_path / "input_no_crs.gpkg")
    input_wgs = str(tmp_path / "input_wgs.gpkg")
    output = str(tmp_path / "output_no_crs.gpkg")

    create_dummy_gpkg(input_no_crs, "feature_no_crs", crs=None)
    create_dummy_gpkg(input_wgs, "feature_wgs", crs="EPSG:4326")

    res = run_merge_script([input_no_crs, input_wgs, output])
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"

    with fiona.open(output) as src:
        # Output CRS should be None or empty
        assert src.crs == {}
        features = list(src)
        assert len(features) == 2
