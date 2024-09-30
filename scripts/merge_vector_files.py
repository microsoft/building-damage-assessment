# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for merging a list of vector files into a single file.

TODO: Check to see if this works if the properties of the files are different, or if
they have different geometry types, or if they have different coordinate reference
systems, etc.
"""

import argparse
import os
import subprocess
import sys


def main():
    """Main function for the merge_vector_files.py script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_files", nargs="+", help="Input vector files to merge")
    parser.add_argument("output_file", help="Output GeoPackage file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )

    args = parser.parse_args()

    input_fns = args.input_files
    output_fn = args.output_file

    assert output_fn.endswith(".gpkg")

    if len(input_fns) < 2:
        print("ERROR: Must provide at least two input files to merge")
        sys.exit(1)

    if os.path.exists(output_fn) and not args.overwrite:
        print(f"ERROR: Output file already exists: {output_fn}, use `--overwrite`")
        sys.exit(1)
    elif os.path.exists(output_fn) and args.overwrite:
        print(f"WARNING: Overwriting output file: {output_fn}")
        os.remove(args.output_fn)

    output_dir = os.path.dirname(output_fn)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        initial_input = input_fns.pop(0)
        subprocess.run(
            ["ogr2ogr", "-f", "GPKG", "-nln", "out", output_fn, initial_input],
            check=True,
        )

        # Merge the remaining input files into the output file
        for input_file in input_fns:
            subprocess.run(
                [
                    "ogr2ogr",
                    "-f",
                    "GPKG",
                    "-update",
                    "-append",
                    "-nln",
                    "out",
                    output_fn,
                    input_file,
                ],
                check=True,
            )

        print(f"Successfully merged {len(input_fns) + 1} files into {output_fn}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
