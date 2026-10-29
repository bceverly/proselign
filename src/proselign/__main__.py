# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Allow ``python -m proselign``."""

import sys

from proselign.cli import main

sys.exit(main())
