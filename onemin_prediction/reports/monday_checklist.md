# Monday Session Checklist

## Pre-open
- Confirm futures sidecar is running and futures features vary bar-to-bar.
- Confirm policy move head exists: trained_models/production/policy_move.json.
- Start session with current recommended env tweaks (if BUY remains suppressed).

## During session (first 60–90 min)
- Check intent mix (BUY/SELL/FLAT) and tradeable rate.
- Watch top gates: no_direction, rv_atr_low, htf_veto.
- If BUY intent stays near zero while BUY labels appear, apply the follow-up tweaks (one at a time).

## Post-session analysis
- Generate bucketization with extended columns:
  - analysis/jan10_bucketized_ext.csv (or current date file)
- Update daily comparison report:
  - reports/daily_comparison.md
- Compare against Jan8/Jan9 baseline in the report.

## Calibration (out-of-sample)
- Score the new live session log:
  - offline_score_trainlog.py -> data/train_log_v3_live_scored.jsonl
- Build tradeable-only log:
  - data/train_log_v3_live_scored_tradeable.jsonl
- Calibrate on this next-session data:
  - calibrator.py -> calib_buy.json / calib_sell.json

## Commands (post-session)
/home/hanumanth/Documents/pyvirtualenv/venv/bin/python offline_score_trainlog.py \
  --in data/train_log_v3_live.jsonl \
  --out data/train_log_v3_live_scored.jsonl --overwrite --force

/home/hanumanth/Documents/pyvirtualenv/venv/bin/python - <<'PY'
import json
from pathlib import Path

inp = Path("data/train_log_v3_live_scored.jsonl")
out = Path("data/train_log_v3_live_scored_tradeable.jsonl")
count = 0
with inp.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("teacher_tradeable"):
            f_out.write(json.dumps(rec, ensure_ascii=True) + "\n")
            count += 1
print(f"wrote {count} tradeable rows -> {out}")
PY

CALIB_MIN_ROWS=50 /home/hanumanth/Documents/pyvirtualenv/venv/bin/python calibrator.py \
  --log data/train_log_v3_live_scored_tradeable.jsonl \
  --buy trained_models/production/calib_buy.json \
  --sell trained_models/production/calib_sell.json
