# Copyright (c) 2024 Overture Maps
# Licensed under the MIT License.
# Code from: https://github.com/OvertureMaps/overturemaps-py/blob/0fad53bceb955b14ac069ef321cbc2486996d5c7/overturemaps/core.py
# Modified to read from Azure Blob Storage instead of S3

from typing import List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs
import fsspec

# Allows for optional import of additional dependencies
try:
    import geopandas as gpd
    from geopandas import GeoDataFrame

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    GeoDataFrame = None


def record_batch_reader(overture_type, bbox=None) -> Optional[pa.RecordBatchReader]:
    """
    Return a pyarrow RecordBatchReader for the desired bounding box and Azure path
    """
    path = _dataset_path(overture_type)

    if bbox:
        xmin, ymin, xmax, ymax = bbox
        filter = (
            (pc.field("bbox", "xmin") < xmax)
            & (pc.field("bbox", "xmax") > xmin)
            & (pc.field("bbox", "ymin") < ymax)
            & (pc.field("bbox", "ymax") > ymin)
        )
    else:
        filter = None

    t_fs = fsspec.filesystem("az", account_name="overturemapswestus2", anon=True)
    pa_fs = fs.PyFileSystem(fs.FSSpecHandler(t_fs))

    dataset = ds.dataset(path, filesystem=pa_fs)
    batches = dataset.to_batches(filter=filter)

    # to_batches() can yield many batches with no rows. I've seen
    # this cause downstream crashes or other negative effects. For
    # example, the ParquetWriter will emit an empty row group for
    # each one bloating the size of a parquet file. Just omit
    # them so the RecordBatchReader only has non-empty ones. Use
    # the generator syntax so the batches are streamed out
    non_empty_batches = (b for b in batches if b.num_rows > 0)

    geoarrow_schema = geoarrow_schema_adapter(dataset.schema)
    reader = pa.RecordBatchReader.from_batches(geoarrow_schema, non_empty_batches)
    return reader


def geodataframe(
    overture_type: str, bbox: (float, float, float, float) = None
) -> GeoDataFrame:
    """
    Loads geoparquet for specified type into a geopandas dataframe

    Parameters
    ----------
    overture_type: type to load
    bbox: optional bounding box for data fetch (xmin, ymin, xmax, ymax)

    Returns
    -------
    GeoDataFrame with the optionally filtered theme data

    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required to use this function")

    reader = record_batch_reader(overture_type, bbox)
    return gpd.GeoDataFrame.from_arrow(reader)


def geoarrow_schema_adapter(schema: pa.Schema) -> pa.Schema:
    """
    Convert a geoarrow-compatible schema to a proper geoarrow schema

    This assumes there is a single "geometry" column with WKB formatting

    Parameters
    ----------
    schema: pa.Schema

    Returns
    -------
    pa.Schema
    A copy of the input schema with the geometry field replaced with
    a new one with the proper geoarrow ARROW:extension metadata

    """
    geometry_field_index = schema.get_field_index("geometry")
    geometry_field = schema.field(geometry_field_index)
    geoarrow_geometry_field = geometry_field.with_metadata(
        {b"ARROW:extension:name": b"geoarrow.wkb"}
    )

    geoarrow_schema = schema.set(geometry_field_index, geoarrow_geometry_field)

    return geoarrow_schema


type_theme_map = {
    "address": "addresses",
    "bathymetry": "base",
    "building": "buildings",
    "building_part": "buildings",
    "division": "divisions",
    "division_area": "divisions",
    "division_boundary": "divisions",
    "place": "places",
    "segment": "transportation",
    "connector": "transportation",
    "infrastructure": "base",
    "land": "base",
    "land_cover": "base",
    "land_use": "base",
    "water": "base",
}


def _dataset_path(overture_type: str, release: str = "2025-11-19.0") -> str:
    """
    Returns the Azure blob storage container and path of the Overture dataset to use.
    This assumes overture_type has been validated, e.g. by the CLI
    """
    theme = type_theme_map[overture_type]
    return f"release/{release}/theme={theme}/type={overture_type}/"


def get_all_overture_types() -> List[str]:
    return list(type_theme_map.keys())
