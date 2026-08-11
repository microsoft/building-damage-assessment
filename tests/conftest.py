# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pytest configuration: make the repository root importable (for `import bda`)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
