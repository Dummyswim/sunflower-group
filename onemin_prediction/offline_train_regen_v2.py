#!/usr/bin/env python3
"""
offline_train_regen_v2.py

Offline trainer that trains from TRAIN_LOG_PATH (JSONL recommended) and promotes a
versioned bundle via symlink:

trained_models/
  bundles/<timestamp>/
    xgb_model.json
    neutral_model.pkl
    calibrator.json (optional carry-forward)
    feature_schema_cols.json
    feature_schema_cols.txt
    manifest.json
  production -> bundles/<timestamp>

Env:
- TRAIN_LOG_PATH (default: data/train_log_v2.jsonl)
- OFFLINE_MIN_ROWS (default: 5000)
- TRAIN_MAX_ROWS (default: 200000)
- MODEL_BUNDLES_DIR (default: trained_models/bundles)
- MODEL_PRODUCTION_LINK (default: trained_models/production)
- FEATURE_SCHEMA_COLS_PATH (optional)
- CALIB_PATH (optional carry-forward into bundle)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

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


def _load_schema_cols(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(data, dict):
        cols = data.get("columns") or data.get("feature_names") or data.get("features")
        if isinstance(cols, list) and all(isinstance(x, str) for x in cols) and cols:
            return list(cols)
        # dict mapping feature->idx
        if "columns" in data and isinstance(data["columns"], dict):
            return list(data["columns"].keys())

    if isinstance(data, list) and all(isinstance(x, str) for x in data) and data:
        return list(data)

    return None


def _load_schema_from_env_or_production(production_link: Path) -> Optional[List[str]]:
    # 1) explicit env file
    sp = os.getenv("FEATURE_SCHEMA_COLS_PATH", "").strip()
    if sp:
        cols = _load_schema_cols(Path(sp))
        if cols:
            return cols

    # 2) schema inside current production bundle (if symlink exists)
    try:
        if production_link.exists():
            resolved = production_link.resolve()
            cols = _load_schema_cols(resolved / "feature_schema_cols.json")
            if cols:
                return cols
    except Exception:
        pass

    # 3) schema next to production link (rare)
    cols = _load_schema_cols(production_link.parent / "feature_schema_cols.json")
    if cols:
        return cols

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.getenv("TRAIN_LOG_PATH", "data/train_log_v2.jsonl"))
    ap.add_argument("--min_rows", type=int, default=int(os.getenv("OFFLINE_MIN_ROWS", "5000")))
    ap.add_argument("--max_rows", type=int, default=int(os.getenv("TRAIN_MAX_ROWS", "200000")))
    args = ap.parse_args()

    rows = load_train_log(args.log, max_rows=args.max_rows)
    if len(rows) < args.min_rows:
        raise SystemExit(f"Not enough rows: {len(rows)} < {args.min_rows}")

    production_link = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))
    schema = _load_schema_from_env_or_production(production_link)

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

    # Save artifacts
    xgb_model.save_model(str(bp.xgb_path))
    import joblib
    joblib.dump(neu_model, str(bp.neutral_path))

    # Save schema (atomic)
    atomic_write_json(bp.schema_cols_json, {"columns": list(schema)})
    atomic_write_text(bp.schema_cols_txt, "\n".join(schema))

    # Carry-forward calibrator (optional)
    calib_env = os.getenv("CALIB_PATH", "").strip()
    if calib_env:
        try:
            csrc = Path(calib_env)
            if csrc.exists():
                bp.calib_path.write_bytes(csrc.read_bytes())
        except Exception:
            pass

    # Validate feature counts match schema
    ok, reason = validate_bundle_feature_counts(bp.xgb_path, list(schema), neutral_path=bp.neutral_path)
    if not ok:
        raise SystemExit(f"bundle validation failed: {reason}")

    # Manifest
    write_manifest(
        bp.manifest_path,
        schema_cols=list(schema),
        notes={
            "trainer": "offline_train_regen_v2",
            "rows": len(rows),
            "dir_rows": int(dir_data.y.size),
            "neu_rows": int(neu_data.y.size),
        },
    )

    # Promote symlink atomically
    promote_bundle_symlink(bundle_dir, production_link)
    print("[OFFLINE] promoted bundle:", bundle_dir)
    print("[OFFLINE] production ->", production_link.resolve())


if __name__ == "__main__":
    main()
