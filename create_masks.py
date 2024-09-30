# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for creating segmentation masks from geojson labels and images.

NOTE: Even though there exists an argument `--buffer_in_meters`, there are no checks to
ensure that the input imagery is in a projected coordinate system (i.e. meters). If the
input imagery is in a geographic coordinate system, the buffer will be in degrees, which
is likely not what you want!
"""

import argparse
import os
import shutil
import subprocess

import cv2
import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.mask
import shapely.geometry

from bda.config import get_args


def add_create_masks_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds the arguments for the create_masks.py script to the base parser."""
    parser.add_argument(
        "--labels.fn",
        type=str,
        help="Path to GeoJSON file containing polygon labels (output from the labeling tool)",
    )
    parser.add_argument(
        "--imagery.raw_fn",
        type=str,
        help="Path to raw input imagery as a COG (cloud-optimized GeoTIFF)",
    )
    parser.add_argument(
        "--experiment_dir", type=str, help="Directory to write dataset to"
    )
    parser.add_argument(
        "--labels.classes", nargs="+", type=str, help="List of class names"
    )
    parser.add_argument(
        "--labels.buffer_in_meters", type=int, help="Buffer in meters around labels"
    )
    parser.add_argument(
        "--labels.class_to_buffer", type=str, help="Class name to buffer"
    )
    parser.add_argument(
        "--labels.class_to_buffer_by",
        type=str,
        help="Class name to set buffered pixels to",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output dataset if it already exists",
    )

    return parser


def get_class_names_from_labels(labels_fn: str, key: str = "class") -> set:
    """Get the class names from a GeoJSON file.

    Args:
        labels_fn (str): Path to GeoJSON file containing polygon labels (output from the
            labeling tool).
        key (str): The key in the GeoJSON file to use for the class names.

    Returns:
        set: Set of class names.
    """
    class_names = set()
    with fiona.open(labels_fn) as f:
        for feature in f:
            class_names.add(feature["properties"][key])
    return class_names


def main() -> None:
    """Main function for the create_masks.py script."""
    args = get_args(description=__doc__, add_extra_parser=add_create_masks_parser)

    input_label_fn = args["labels"]["fn"]
    input_image_fn = args["imagery"]["raw_fn"]
    output_dir = args["experiment_dir"]
    class_names = args["labels"]["classes"]
    buffer_in_meters = args["labels"]["buffer_in_meters"]
    class_to_buffer = args["labels"]["class_to_buffer"]
    class_to_buffer_by = args["labels"]["class_to_buffer_by"]
    overwrite = args["overwrite"]

    # we include +1 as we use 0 as a "not labeled" class by convention
    class_name_to_idx_map = {
        class_name: idx + 1 for idx, class_name in enumerate(class_names)
    }

    if set(class_names) != get_class_names_from_labels(input_label_fn):
        print(
            "WARNING: The class names in the config file do not match the class names"
            + " in the input label file."
        )

    assert os.path.exists(input_label_fn)
    assert input_label_fn.endswith(".geojson")
    assert os.path.exists(input_image_fn)
    assert input_image_fn.endswith(".tif")

    name = os.path.basename(input_image_fn).replace(".tif", "")

    output_mask_fn = os.path.join(output_dir, f"{name}_mask.tif")
    output_warped_label_fn = os.path.join(output_dir, f"{name}_labels_warped.geojson")
    output_cropped_image_fn = os.path.join(output_dir, "images", f"{name}_cropped.tif")
    output_buffered_mask_fn = os.path.join(output_dir, "masks", f"{name}_buffered.tif")
    all_output_fns = [
        output_mask_fn,
        output_warped_label_fn,
        output_cropped_image_fn,
        output_buffered_mask_fn,
    ]

    # Check if any of the output files already exist
    if any([os.path.exists(fn) for fn in all_output_fns]) and not overwrite:
        print("Output files already exist, use --overwrite to overwrite them. Exiting.")
        return

    # Make sure the output directories exist
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Make a backup of the input label file
    shutil.copy(input_label_fn, os.path.join(output_dir, "labels"))

    ##########
    # Load information about the input image
    with rasterio.open(input_image_fn) as f:
        profile = f.profile
        dst_crs = f.crs.to_string()

    ##########
    # Warp the labels to the CRS of the input image
    command = [
        "ogr2ogr",
        "-f",
        "GeoJSON",
        "-t_srs",
        dst_crs,
        output_warped_label_fn,
        input_label_fn,
    ]
    assert subprocess.call(command) == 0

    ##########
    # Crop the input image to the extent of the labels
    with fiona.open(input_label_fn) as f:
        geom = shapely.geometry.mapping(shapely.geometry.box(*f.bounds))
    geom = dict(fiona.transform.transform_geom("epsg:4326", dst_crs, geom))
    del geom["geometries"]
    geom = shapely.geometry.mapping(shapely.geometry.shape(geom).envelope)

    with rasterio.open(input_image_fn) as f:
        data, transform = rasterio.mask.mask(f, [geom], crop=True)

    _, height, width = data.shape

    profile["height"] = height
    profile["width"] = width
    profile["transform"] = transform
    profile["predictor"] = 2
    with rasterio.open(output_cropped_image_fn, "w", **profile) as f:
        f.write(data)

    ##########
    # Create mask
    with rasterio.open(output_cropped_image_fn) as f:
        profile = f.profile
        left, bottom, right, top = f.bounds
        width = f.width
        height = f.height
        dst_crs = f.crs.to_string()

    command = [
        "gdal_rasterize",
        "-q",  # be quiet about it
        "-ot",
        "Byte",  # the output dtype of the raster should be uint8
        "-a_nodata",
        "0",  # the nodata value should be "0", this value will represent not-labeled in our training process
        "-init",
        "0",  # initialize all values to 0
        "-burn",
        str(
            class_name_to_idx_map[class_names[0]]
        ),  # we will burn in the first class value to all polygons in the GeoJSON that match the first class label
        "-of",
        "GTiff",  # the output should be a GeoTIFF
        "-co",
        "TILED=YES",  # the output should be tiled, similar to COGs -- https://www.cogeo.org/ -- this is important for fast windowed reads
        "-co",
        "BLOCKXSIZE=512",  # this is important for fast windowed reads
        "-co",
        "BLOCKYSIZE=512",  # this is important for fast windowed reads
        "-co",
        "INTERLEAVE=PIXEL",  # this is important for fast windowed reads
        "-where",
        f"class='{class_names[0]}'",  # burn in values for polygons where the class label is the first class label
        "-te",
        str(left),
        str(bottom),
        str(right),
        str(top),  # the output GeoTIFF should cover the same bounds as the input image
        "-ts",
        str(width),
        str(
            height
        ),  # the output GeoTIFF should have the same height and width as the input image
        "-co",
        "COMPRESS=LZW",  # compress it
        "-co",
        "PREDICTOR=2",  # compress it good
        "-co",
        "BIGTIFF=YES",  # just incase the image is bigger than 4GB
        output_warped_label_fn,
        output_mask_fn,
    ]
    assert subprocess.call(command) == 0

    for i in range(1, len(class_names)):
        command = [
            "gdal_rasterize",
            "-q",
            "-b",
            "1",
            "-burn",
            str(class_name_to_idx_map[class_names[i]]),
            "-where",
            f"class='{class_names[i]}'",
            input_label_fn,
            output_mask_fn,
        ]
        assert subprocess.call(command) == 0

    ##########
    # Buffer mask around buildings
    with rasterio.open(output_mask_fn) as f:
        mask = f.read().squeeze()
        mask_profile = f.profile

    nodata_mask = (mask != class_name_to_idx_map[class_to_buffer]).astype(np.uint8)
    transform = cv2.distanceTransform(nodata_mask, distanceType=cv2.DIST_L2, maskSize=3)
    background_mask = (transform > 0) & (transform < buffer_in_meters)
    mask[background_mask] = class_name_to_idx_map[class_to_buffer_by]

    with rasterio.open(output_buffered_mask_fn, "w", **mask_profile) as f:
        f.write(mask, 1)

    ##########
    # Check that the buffered mask and the cropped image have the same dimensions
    with rasterio.open(output_cropped_image_fn) as f:
        t_height, t_width = f.shape
    with rasterio.open(output_buffered_mask_fn):
        assert f.shape[0] == t_height
        assert f.shape[1] == t_width

    os.remove(output_warped_label_fn)
    os.remove(output_mask_fn)


if __name__ == "__main__":
    main()
