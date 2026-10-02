# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Exceptions raised by the JAX backend."""


class UnsupportedArchitectureError(Exception):
    """The model's architecture (or one of its structural features) is not supported."""


class UnsupportedCheckpointError(Exception):
    """
    The checkpoint cannot be loaded although its architecture is supported,
    for example because it is pre-quantised or its tensor layout is unexpected.
    """


class DeviceMemoryError(Exception):
    """A computation or the model parameters would not fit into device memory."""
