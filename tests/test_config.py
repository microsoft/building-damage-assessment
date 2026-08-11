# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the GPU config helpers in ``bda.config``."""

from bda.config import normalize_gpu_ids


# ---------------------------------------------------------------------------
# normalize_gpu_ids
# ---------------------------------------------------------------------------
def test_normalize_list():
    assert normalize_gpu_ids([0, 1, 2]) == [0, 1, 2]


def test_normalize_single_int():
    assert normalize_gpu_ids(2) == [2]


def test_normalize_single_gpu_id_fallback():
    assert normalize_gpu_ids(None, 3) == [3]


def test_normalize_precedence_gpu_ids_over_gpu_id():
    assert normalize_gpu_ids([0, 1], 5) == [0, 1]


def test_normalize_comma_string():
    assert normalize_gpu_ids("0,1,2") == [0, 1, 2]


def test_normalize_space_string():
    assert normalize_gpu_ids("0 1 2") == [0, 1, 2]


def test_normalize_dedupe_preserves_order():
    assert normalize_gpu_ids([0, 0, 1, 1, 2]) == [0, 1, 2]


def test_normalize_none_is_empty():
    assert normalize_gpu_ids(None, None) == []
