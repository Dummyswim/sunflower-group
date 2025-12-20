import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_pipeline_regen_v2 import AdaptiveModelPipeline


class _DummyXGB:
    pass


def _write_calib(path, *, a=0.3, b=0.0, n=0, auc=0.6, inverted=False):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "a": float(a),
                "b": float(b),
                "n": int(n),
                "auc": float(auc),
                "inverted": bool(inverted),
            },
            f,
        )


def test_calibrator_bypass_latch_resets():
    old_env = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "calibrator.json")
            os.environ["CALIB_PATH"] = path
            os.environ["CALIB_MIN_N"] = "200"

            _write_calib(path, n=50)
            pipe = AdaptiveModelPipeline(_DummyXGB())
            assert pipe._calib_bypass is True

            _write_calib(path, n=500)
            ok = pipe.reload_calibration(path)
            assert ok is True
            assert pipe._calib_bypass is False
    finally:
        os.environ.clear()
        os.environ.update(old_env)
