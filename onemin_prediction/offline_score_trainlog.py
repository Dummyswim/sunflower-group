#!/usr/bin/env python3
"""
Offline scoring pass for SignalContext logs.

Adds model.p_success_raw / model.p_success_calib using the current policy bundle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from policy_pipeline import load_policy_models
from signal_context import compose_policy_features


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v == v and v not in (float("inf"), float("-inf")):
            return v
    except Exception:
        pass
    return float(default)


def _should_score(rec: Dict[str, Any], force: bool) -> bool:
    if force:
        return True
    model = rec.get("model")
    if isinstance(model, dict):
        if model.get("p_success_raw") is not None:
            return False
        if model.get("policy_success_raw") is not None:
            return False
    return True


def _score_record(
    rec: Dict[str, Any],
    *,
    policy_cols: list[str],
    pipe,
) -> Optional[Dict[str, Any]]:
    teacher_dir = str(rec.get("teacher_dir", "")).upper()
    if teacher_dir not in ("BUY", "SELL"):
        return None

    feats = rec.get("features") or {}
    rule_signals = rec.get("rule_signals") or {}
    gates = rec.get("gates") or {}
    teacher_strength = _safe_float(rec.get("teacher_strength", 0.0), 0.0)

    policy_feats = compose_policy_features(
        features=feats,
        rule_signals=rule_signals,
        gates=gates,
        teacher_strength=teacher_strength,
    )
    values = [_safe_float(policy_feats.get(c, 0.0), 0.0) for c in policy_cols]
    p_raw, p_cal = pipe.predict_success(
        feature_names=policy_cols,
        feature_values=values,
        teacher_dir=teacher_dir,
    )
    if p_raw is None:
        return None
    return {"p_success_raw": p_raw, "p_success_calib": p_cal}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.getenv("TRAIN_LOG_PATH", "data/train_log_v3_canonical.jsonl"))
    ap.add_argument("--out", dest="out", default="data/train_log_v3_canonical_scored.jsonl")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-score even if model.p_success_raw exists")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = no limit")
    ap.add_argument("--buy_model", default=os.getenv("POLICY_BUY_PATH", "trained_models/production/policy_buy.json"))
    ap.add_argument("--sell_model", default=os.getenv("POLICY_SELL_PATH", "trained_models/production/policy_sell.json"))
    ap.add_argument("--schema", default=os.getenv("POLICY_SCHEMA_COLS_PATH", "trained_models/production/policy_schema_cols.json"))
    ap.add_argument("--calib_buy", default=os.getenv("CALIB_BUY_PATH", "trained_models/production/calib_buy.json"))
    ap.add_argument("--calib_sell", default=os.getenv("CALIB_SELL_PATH", "trained_models/production/calib_sell.json"))
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    if not inp.exists():
        print(f"[SCORE] input not found: {inp}", file=sys.stderr)
        return 2
    if out.exists() and not args.overwrite:
        print(f"[SCORE] output exists (use --overwrite): {out}", file=sys.stderr)
        return 2
    if inp.resolve() == out.resolve():
        print("[SCORE] input and output must be different files", file=sys.stderr)
        return 2

    pipe = load_policy_models(
        buy_path=str(args.buy_model),
        sell_path=str(args.sell_model),
        schema_path=str(args.schema),
        calib_buy_path=str(args.calib_buy) if args.calib_buy else None,
        calib_sell_path=str(args.calib_sell) if args.calib_sell else None,
    )
    policy_cols = list(pipe.feature_schema_names)

    tmp_path = out.with_suffix(out.suffix + ".tmp")
    total = 0
    scored = 0
    skipped = 0
    bad = 0
    started = time.time()

    with open(inp, "r", encoding="utf-8", errors="ignore") as f_in, open(tmp_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if args.max_rows and total >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                continue

            if _should_score(rec, args.force):
                upd = _score_record(rec, policy_cols=policy_cols, pipe=pipe)
                if upd is not None:
                    model = rec.get("model")
                    if not isinstance(model, dict):
                        model = {}
                    model["p_success_raw"] = upd["p_success_raw"]
                    model["p_success_calib"] = upd["p_success_calib"]
                    rec["model"] = model
                    prov = rec.get("provenance")
                    if not isinstance(prov, dict):
                        prov = {}
                    prov["scored"] = True
                    rec["provenance"] = prov
                    scored += 1
                else:
                    skipped += 1
            else:
                skipped += 1

            f_out.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=True) + "\n")

            if total % 20000 == 0:
                elapsed = time.time() - started
                print(f"[SCORE] rows={total} scored={scored} skipped={skipped} bad={bad} elapsed={elapsed:.1f}s")

    os.replace(tmp_path, out)
    elapsed = time.time() - started
    print(f"[SCORE] done rows={total} scored={scored} skipped={skipped} bad={bad} elapsed={elapsed:.1f}s")
    print(f"[SCORE] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
