# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Methods to handle the parsing and merging of command line and YAML file arguments."""

import argparse
from typing import Callable, Optional, Sequence, Union

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
        "use_constraint_loss": bool,
    },
    "inference": {
        "output_subdir": str,
        "batch_size": int,
        "gpu_id": int,
        "checkpoint_fn": str,
    },
}


def normalize_gpu_ids(
    gpu_ids: Optional[Union[Sequence[int], str, int]] = None,
    gpu_id: Optional[int] = None,
) -> list[int]:
    """Normalize a GPU specification into an ordered, de-duplicated list of ids.

    Precedence: an explicit ``gpu_ids`` (list/tuple, or a comma/space separated
    string) takes priority over a single ``gpu_id``. Returns an empty list when
    neither is provided (i.e. CPU).

    Args:
        gpu_ids: A list/tuple of ints, a comma/space-separated string (e.g.
            ``"0,1,2"``), a single int, or ``None``.
        gpu_id: A single GPU id, used only when ``gpu_ids`` is ``None``.

    Returns:
        Ordered list of unique GPU ids.
    """
    raw: list = []
    if gpu_ids is not None:
        if isinstance(gpu_ids, str):
            raw = gpu_ids.replace(",", " ").split()
        elif isinstance(gpu_ids, int):
            raw = [gpu_ids]
        else:
            raw = list(gpu_ids)
    elif gpu_id is not None:
        raw = [gpu_id]

    ids: list[int] = []
    for item in raw:
        value = int(item)
        if value not in ids:
            ids.append(value)
    return ids


def resolve_clip_range(
    imagery_config: dict,
) -> Optional[tuple[float, float]]:
    """Resolve the imagery clip range from an ``imagery`` config block.

    Both ``fine_tune.py`` and ``inference.py`` call this so that training and
    inference always preprocess imagery identically.

    Precedence: ``no_clip`` disables clipping outright, otherwise an explicit
    ``clip_range`` is used. A missing key defaults to ``(0, 1)``, which is the
    behavior from before this was configurable; an explicit ``clip_range: null``
    disables clipping.

    Args:
        imagery_config (dict): The ``imagery`` block of a config dictionary.

    Returns:
        Optional[tuple[float, float]]: The ``(low, high)`` bounds, or ``None``
            when clipping is disabled.
    """
    if imagery_config.get("no_clip"):
        return None
    clip_range = imagery_config.get("clip_range", (0, 1))
    if clip_range is None:
        return None
    return (float(clip_range[0]), float(clip_range[1]))


def add_clip_range_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the imagery clipping flags shared by fine_tune.py and inference.py.

    These are registered on both scripts so that the same preprocessing can be
    requested at training and inference time.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments to.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    parser.add_argument(
        "--imagery.clip_range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Clip the normalized imagery to these bounds. Defaults to `0 1`,"
        + " which suits min/max style normalization values but flattens every"
        + " below-mean pixel when using true mean/std standardization. Use e.g."
        + " `-3 3` for standardized imagery, or --imagery.no_clip to disable.",
    )
    parser.add_argument(
        "--imagery.no_clip",
        action="store_true",
        default=None,  # keep None so it doesn't override the config file
        help="Disable clipping of the normalized imagery entirely.",
    )
    return parser


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
