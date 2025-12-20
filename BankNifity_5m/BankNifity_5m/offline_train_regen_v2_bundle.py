#!/usr/bin/env python3
# offline_train_regen_v2_bundle.py
"""
Offline trainer (v3) that trains from TRAIN_LOG_PATH (JSONL preferred) and
promotes a versioned bundle via symlink.

Usage:
  python offline_train_regen_v3.py \
    --log data/train_log_v2.jsonl \
    --min_rows 5000

Env:
- MODEL_BUNDLES_DIR (default: trained_models/bundles)
- MODEL_PRODUCTION_LINK (default: trained_models/production)
- FEATURE_SCHEMA_COLS_PATH (optional; otherwise inferred from log)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_bundle import (
    make_bundle_dir,
    bundle_paths,
    atomic_write_json,
    atomic_write_text,
    write_manifest,
    validate_bundle_feature_counts,
    promote_bundle_symlink,
)

from online_trainer_regen_v2 import (
    load_train_log,
    build_directional_dataset,
    build_neutral_dataset,
    train_xgb_dir,
    train_neutral,
)


def _load_schema_from_env_or_disk(xgb_out_dir: Path) -> Optional[List[str]]:
    sp = os.getenv("FEATURE_SCHEMA_COLS_PATH", "").strip()
    if sp:
        p = Path(sp)
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            cols = obj.get("columns") or obj.get("feature_names") or obj.get("features")
            if isinstance(cols, list) and cols:
                return list(cols)
    # fallback: schema next to production link
    p2 = xgb_out_dir / "feature_schema_cols.json"
    if p2.exists():
        obj = json.loads(p2.read_text(encoding="utf-8"))
        cols = obj.get("columns") or obj.get("feature_names") or obj.get("features")
        if isinstance(cols, list) and cols:
            return list(cols)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.getenv("TRAIN_LOG_PATH", "data/train_log_v2.jsonl"))
    ap.add_argument("--min_rows", type=int, default=int(os.getenv("OFFLINE_MIN_ROWS", "5000")))
    ap.add_argument("--max_rows", type=int, default=int(os.getenv("TRAIN_MAX_ROWS", "200000")))
    args = ap.parse_args()

    rows = load_train_log(args.log, max_rows=args.max_rows)
    if len(rows) < args.min_rows:
        raise SystemExit(f"Not enough rows: {len(rows)} < {args.min_rows}")

    production_link = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))
    # production link parent is trained_models; schema may live in active bundle
    schema = _load_schema_from_env_or_disk(production_link.resolve() if production_link.exists() else production_link.parent)
    if not schema:
        feats = (rows[-1].get("features") or {})
        schema = [k for k in feats.keys() if k != "aux_ret_main"]
        schema.sort()

    dir_data = build_directional_dataset(rows, schema_cols=schema)
    neu_data = build_neutral_dataset(rows, schema_cols=schema)

    print(f"[OFFLINE] rows={len(rows)} dir={dir_data.y.size} neu={neu_data.y.size} cols={len(schema)}")

    xgb_model = train_xgb_dir(dir_data)
    neu_model = train_neutral(neu_data)
    if xgb_model is None or neu_model is None:
        raise SystemExit("training failed (model None)")

    bundles_dir = Path(os.getenv("MODEL_BUNDLES_DIR", "trained_models/bundles"))
    bundle_dir = make_bundle_dir(bundles_dir)
    bp = bundle_paths(bundle_dir)

    xgb_model.save_model(str(bp.xgb_path))

    import joblib
    joblib.dump(neu_model, str(bp.neutral_path))

    atomic_write_json(bp.schema_cols_json, {"columns": list(schema)})
    atomic_write_text(bp.schema_cols_txt, "\n".join(schema))

    # carry forward calibrator if exists
    calib_env = os.getenv("CALIB_PATH", "").strip()
    if calib_env:
        try:
            csrc = Path(calib_env)
            if csrc.exists():
                bp.calib_path.write_bytes(csrc.read_bytes())
        except Exception:
            pass

    ok, reason = validate_bundle_feature_counts(bp.xgb_path, list(schema), neutral_path=bp.neutral_path)
    if not ok:
        raise SystemExit(f"bundle validation failed: {reason}")

    write_manifest(
        bp.manifest_path,
        schema_cols=list(schema),
        notes={
            "trainer": "offline_train_regen_v2_bundle",
            "rows": len(rows),
            "dir_rows": int(dir_data.y.size),
            "neu_rows": int(neu_data.y.size),
        },
    )

    promote_bundle_symlink(bundle_dir, production_link)
    print("[OFFLINE] promoted bundle:", bundle_dir)
    print("[OFFLINE] production ->", production_link.resolve())


if __name__ == "__main__":
    main()
