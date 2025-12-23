#!/usr/bin/env python3
"""
Offline trainer for policy BUY/SELL models using SignalContext logs.

Promotes a versioned bundle via symlink:
trained_models/
  bundles/<timestamp>/
    policy_buy.json
    policy_sell.json
    calib_buy.json (optional carry-forward)
    calib_sell.json (optional carry-forward)
    policy_schema_cols.json
    policy_schema_cols.txt
    feature_schema_cols.json
    feature_schema_cols.txt
    manifest.json
  production -> bundles/<timestamp>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from eval_before_promotion import evaluate_holdout
from model_bundle import (
    make_bundle_dir,
    bundle_paths,
    atomic_write_json,
    atomic_write_text,
    write_manifest,
    validate_bundle_feature_counts,
    promote_bundle_symlink,
)
from online_trainer_regen_v2_bundle import build_policy_dataset, train_policy_model
from pretrain_validator import validate_pretrain
from schema_contract import SchemaResolutionError
from signal_log_utils import load_signal_log, summarize_sources, data_range
from signal_context import compose_policy_schema


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
        if "columns" in data and isinstance(data["columns"], dict):
            return list(data["columns"].keys())

    if isinstance(data, list) and all(isinstance(x, str) for x in data) and data:
        return list(data)

    return None


def _load_base_schema(production_link: Path) -> Optional[List[str]]:
    sp = os.getenv("FEATURE_SCHEMA_COLS_PATH", "").strip()
    if sp:
        cols = _load_schema_cols(Path(sp))
        if cols:
            return cols

    cols = _load_schema_cols(Path("data/feature_schema_cols.json"))
    if cols:
        return cols

    try:
        if production_link.exists():
            resolved = production_link.resolve()
            cols = _load_schema_cols(resolved / "feature_schema_cols.json")
            if cols:
                return cols
    except Exception:
        pass

    cols = _load_schema_cols(production_link.parent / "feature_schema_cols.json")
    if cols:
        return cols

    return None


def _load_policy_schema(production_link: Path) -> Optional[List[str]]:
    sp = os.getenv("POLICY_SCHEMA_COLS_PATH", "").strip()
    if sp:
        cols = _load_schema_cols(Path(sp))
        if cols:
            return cols

    try:
        if production_link.exists():
            resolved = production_link.resolve()
            cols = _load_schema_cols(resolved / "policy_schema_cols.json")
            if cols:
                return cols
    except Exception:
        pass

    cols = _load_schema_cols(production_link.parent / "policy_schema_cols.json")
    if cols:
        return cols

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.getenv("TRAIN_LOG_PATH", ""))
    ap.add_argument("--min_rows", type=int, default=int(os.getenv("OFFLINE_MIN_ROWS", "5000")))
    ap.add_argument("--max_rows", type=int, default=int(os.getenv("TRAIN_MAX_ROWS", "200000")))
    args = ap.parse_args()

    production_link = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))
    base_schema = _load_base_schema(production_link)
    if not base_schema:
        raise SchemaResolutionError("FEATURE_SCHEMA_COLS_PATH missing or invalid; base schema required")

    policy_schema = _load_policy_schema(production_link)
    if not policy_schema:
        policy_schema = compose_policy_schema(list(base_schema))

    if not args.log:
        raise SystemExit("TRAIN_LOG_PATH missing; --log is required")

    rows = load_signal_log(args.log, max_rows=args.max_rows, schema_cols=list(base_schema))
    if len(rows) < args.min_rows:
        raise SystemExit(f"Not enough rows: {len(rows)} < {args.min_rows}")

    ok, report = validate_pretrain(rows, schema_cols=list(policy_schema))
    if not ok:
        raise SystemExit(f"pretrain validation failed: {report.get('reasons')}")

    buy_data = build_policy_dataset(rows, schema_cols=policy_schema, teacher_dir="BUY")
    sell_data = build_policy_dataset(rows, schema_cols=policy_schema, teacher_dir="SELL")

    print(
        f"[OFFLINE] rows={len(rows)} buy_rows={buy_data.n_rows} sell_rows={sell_data.n_rows} cols={len(policy_schema)}"
    )

    buy_model = train_policy_model(buy_data)
    sell_model = train_policy_model(sell_data)
    if buy_model is None or sell_model is None:
        raise SystemExit("training failed (model None)")

    eval_ok, eval_report = evaluate_holdout(rows, list(policy_schema), buy_model, sell_model)
    if not eval_ok:
        raise SystemExit(f"eval gate failed: {eval_report}")

    bundles_dir = Path(os.getenv("MODEL_BUNDLES_DIR", "trained_models/bundles"))
    bundle_dir = make_bundle_dir(bundles_dir)
    bp = bundle_paths(bundle_dir)

    buy_model.save_model(str(bp.policy_buy_path))
    sell_model.save_model(str(bp.policy_sell_path))

    atomic_write_json(bp.schema_cols_json, {"columns": list(policy_schema)})
    atomic_write_text(bp.schema_cols_txt, "\n".join(policy_schema))
    atomic_write_json(bp.base_schema_cols_json, {"columns": list(base_schema)})
    atomic_write_text(bp.base_schema_cols_txt, "\n".join(base_schema))

    calib_buy = os.getenv("CALIB_BUY_PATH", "").strip()
    calib_sell = os.getenv("CALIB_SELL_PATH", "").strip()
    if calib_buy:
        try:
            csrc = Path(calib_buy)
            if csrc.exists():
                bp.calib_buy_path.write_bytes(csrc.read_bytes())
        except Exception:
            pass
    if calib_sell:
        try:
            csrc = Path(calib_sell)
            if csrc.exists():
                bp.calib_sell_path.write_bytes(csrc.read_bytes())
        except Exception:
            pass

    ok, reason = validate_bundle_feature_counts(
        bp.policy_buy_path,
        bp.policy_sell_path,
        list(policy_schema),
    )
    if not ok:
        raise SystemExit(f"bundle validation failed: {reason}")

    write_manifest(
        bp.manifest_path,
        schema_cols=list(policy_schema),
        notes={
            "trainer": "offline_train_policy",
            "rows": len(rows),
            "buy_rows": int(buy_data.n_rows),
            "sell_rows": int(sell_data.n_rows),
            "pretrain_report": report,
            "eval_report": eval_report,
        },
        data_range=data_range(rows),
        source_counts=summarize_sources(rows),
        validation_report={"pretrain": report, "eval": eval_report},
    )

    promote_bundle_symlink(bundle_dir, production_link)
    print("[OFFLINE] promoted bundle:", bundle_dir)
    print("[OFFLINE] production ->", production_link.resolve())


if __name__ == "__main__":
    main()
