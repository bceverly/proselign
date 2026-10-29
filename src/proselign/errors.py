# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Exception types, each carrying the process exit status it maps to.

The exit codes are part of the CLI contract (see man/proselign.1): scripts may
branch on them, so a code is never reused for a different meaning.
"""

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_VOICE = 3
EXIT_VALIDATION = 4
EXIT_BACKEND = 5


class ProselignError(Exception):
    """Base class; anything raised as this is reported on stderr, not a traceback."""

    exit_code = EXIT_ERROR


class UsageError(ProselignError):
    """The command line or its inputs are malformed."""

    exit_code = EXIT_USAGE


class VoiceError(ProselignError):
    """A voice is missing, invalid, or cannot be stored."""

    exit_code = EXIT_VOICE


class ValidationError(ProselignError):
    """A rewrite failed fidelity checks and --strict forbids emitting it."""

    exit_code = EXIT_VALIDATION


class BackendError(ProselignError):
    """The inference backend is unreachable or returned an unusable reply."""

    exit_code = EXIT_BACKEND
