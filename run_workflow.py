# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Helper script for running the entire workflow."""

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    subprocess.run(["python", "create_masks.py", "--config", args.config])
    subprocess.run(["python", "fine_tune.py", "--config", args.config])
    subprocess.run(["python", "inference.py", "--config", args.config])


if __name__ == "__main__":
    main()
