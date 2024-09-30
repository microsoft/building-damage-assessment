#!/usr/bin/env bash

# Test download_building_footprints.py using a GeoJSON file of the
# Kakuma refugee camp in Turkana County, Kenya
python download_building_footprints.py --source microsoft --input_fn tests/kakuma_test_area.geojson --output_dir tests/ --country_alpha2_iso_code KE
python download_building_footprints.py --source google --input_fn tests/kakuma_test_area.geojson --output_dir tests/ --country_alpha2_iso_code KE
python download_building_footprints.py --source osm --input_fn tests/kakuma_test_area.geojson --output_dir tests/ --country_alpha2_iso_code KE
