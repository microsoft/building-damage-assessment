# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Methods to handle the parsing and merging of command line and YAML file arguments."""

import argparse
from typing import Callable, Optional

import yaml

_DEFAULT_CONFIG = {
    "experiment_name": str,
    "experiment_dir": str,
    "imagery": {
        "raw_fn": str,
        "num_channels": int,
        "normalization_means": list,
        "normalization_stds": list,
    },
    "labels": {
        "fn": str,
        "classes": list,
        "buffer_in_meters": int,
        "class_to_buffer": str,
        "class_to_buffer_by": str,
    },
    "training": {
        "learning_rate": float,
        "max_epochs": int,
        "batch_size": int,
        "gpu_id": int,
        "log_dir": str,
        "checkpoint_subdir": str,
    },
    "inference": {
        "output_subdir": str,
        "batch_size": int,
        "gpu_id": int,
        "checkpoint_fn": str,
    },
}


def _get_base_parser(description: Optional[str]) -> argparse.ArgumentParser:
    """The base argument parser for all scripts.

    Args:
        description (Optional[str]): The description of the script.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser


def _validate_config(config: dict, template: dict = _DEFAULT_CONFIG) -> None:
    """Checks that a loaded config file is valid.

    Args:
        config (dict): The configuration dictionary to validate.

    Raises:
        KeyError: If a key is missing from the config file.
        TypeError: If a value is not of the expected type.
    """
    for key, value in template.items():
        if key not in config:
            raise KeyError(f"Key '{key}' expected, but not found in config file.")

        if isinstance(value, dict):
            _validate_config(config[key], value)
        elif not isinstance(config[key], value):
            raise TypeError(
                f"Key '{key}' is not of type '{value}' (value of '{config[key]}'"
                + " found)."
            )


def _merge_argparse_and_config(config: dict, args: argparse.Namespace) -> dict:
    """Merges the config dictionary loaded by YAML with the argparse namespace.

    Overwrites the values in the config dictionary with any values passed on the
    command line. Note, for nested keys, the command line arguments will have '.' to
    separate the keys, e.g. `--training.learning_rate 0.01`.

    Args:
        config (dict): A configuration dictionary loaded from a YAML file.
        args (argparse.Namespace): Subset of the configuration dictionary loaded
            from the command line.

    Returns:
        dict: The merged configuration dictionary.
    """
    for key, value in vars(args).items():
        if value is not None:
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value

    return config


def get_args(description: Optional[str], add_extra_parser: Optional[Callable]) -> dict:
    """Handles the parsing of all arguments for a script.

    Args:
        description (Optional[str]): The description of the script (this is shown when
            `--help` is passed).
        add_extra_parser (Optional[Callable]): A function that adds extra command line
            arguments to the base parser so that a user can override config file values.

    Returns:
        dict: Merged set of arguments from the config file (passed with `--config`) and
            command line.
    """
    parser = _get_base_parser(description=description)
    if add_extra_parser is not None:
        parser = add_extra_parser(parser)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = _merge_argparse_and_config(config, args)
    _validate_config(config)
    return config
