# Lab Replay Workflow

## 1) Record ticks during live run

Set:

- `STRAT_TICK_LOG_ENABLED=1`
- `STRAT_TICK_LOG_PATH=logs/ticks_{date}.ndjson`

The strategy appends one JSON line per packet with sequence + receive timestamp.

## 2) Replay a session

```bash
python lab/replay_ticks.py \
  --ticks logs/ticks_2026-02-20.ndjson \
  --jsonl-out logs/replay_2026-02-20.jsonl \
  --log-out logs/replay_2026-02-20.log \
  --truncate-output
```

## 3) Diff baseline vs candidate

```bash
python lab/diff_signals.py \
  --baseline logs/strategy_signals_baseline.jsonl \
  --candidate logs/replay_2026-02-20.jsonl \
  --csv-out logs/diff_2026-02-20.csv
```

## 4) Document the run

Update:

- `docs/session_logs/YYYY-MM-DD.md`
- `docs/experiment_registry.csv`
- corresponding ADR under `docs/adr/`
