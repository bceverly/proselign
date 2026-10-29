# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Proselign: learn a writer's prose style and align text to it, locally."""

__version__ = "1.0.0.0"


def full_version() -> str:
    """``__version__`` plus the build stamp ``make install`` writes (``-dev``)."""
    try:
        from proselign._build import BUILD_SUFFIX  # pylint: disable=import-outside-toplevel
    except ImportError:
        return __version__
    return __version__ + BUILD_SUFFIX
