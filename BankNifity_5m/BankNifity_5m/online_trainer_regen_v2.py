#!/usr/bin/env python3
"""
Compatibility shim.

Many modules import `online_trainer_regen_v2`.
Implementation lives in `online_trainer_regen_v2_bundle.py`.

Important: `from module import *` does NOT import underscore names like `_parse_feature_csv`,
so we explicitly re-export it for offline_eval.py and calibrator.py.
"""

from online_trainer_regen_v2_bundle import *  # noqa: F401,F403
from online_trainer_regen_v2_bundle import _parse_feature_csv  # noqa: F401
