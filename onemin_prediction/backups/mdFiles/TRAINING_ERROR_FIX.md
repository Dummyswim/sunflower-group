# TRAINING ERROR FIX - SUMMARY

**Date:** December 7, 2025  
**Issue:** Training script requires additional environment variables  
**Status:** ✅ FIXED

---

## The Problem

When you tried to run the training script, you encountered **two errors in sequence**:

### Error #1: Invalid Arguments
```bash
$ python offline_train_2min.py \
  --start-date 2025-09-01 \
  --end-date 2025-12-07 \
  --output labeled_data.csv

ERROR: TRAIN_START_DATE and TRAIN_END_DATE must be set
```

**Issue:** Script doesn't accept command-line arguments

### Error #2: Missing Output Paths
```bash
$ export TRAIN_START_DATE="2025-09-01"
$ export TRAIN_END_DATE="2025-12-07"
$ python offline_train_2min.py

ERROR: XGB_PATH and NEUTRAL_PATH must be set for offline training outputs
```

**Issue:** Script needs to know WHERE to save the trained models

---

## The Solution

You need **6 environment variables** (not 4):

### Required Variables (Script Fails Without These)

```bash
# Dhan API credentials
export DHAN_ACCESS_TOKEN="your_token_here"
export DHAN_CLIENT_ID="your_client_id_here"

# Training date range
export TRAIN_START_DATE="2025-09-01"
export TRAIN_END_DATE="2025-12-07"

# Model output paths (NEW - REQUIRED)
export XGB_PATH="trained_models/production/xgb_model.pkl"
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl"
```

### Optional Variables (Defaults Used If Not Set)

```bash
export TRADE_HORIZON_MIN=2              # 2 minutes
export TRADE_TP_PCT=0.0015              # 0.15%
export TRADE_SL_PCT=0.0008              # 0.08%
export INTRADAY_CACHE_ENABLE=1          # Enable caching
```

---

## Complete Working Command

**One-liner (Copy & Paste):**

```bash
export DHAN_ACCESS_TOKEN="your_token" && \
export DHAN_CLIENT_ID="your_client_id" && \
export TRAIN_START_DATE="2025-09-01" && \
export TRAIN_END_DATE="2025-12-07" && \
export XGB_PATH="trained_models/production/xgb_model.pkl" && \
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl" && \
cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction && \
python offline_train_2min.py
```

**Multi-line (Easier to understand):**

```bash
# Set API credentials
export DHAN_ACCESS_TOKEN="your_token"
export DHAN_CLIENT_ID="your_client_id"

# Set training dates
export TRAIN_START_DATE="2025-09-01"
export TRAIN_END_DATE="2025-12-07"

# Set model output paths (THE KEY FIX)
export XGB_PATH="trained_models/production/xgb_model.pkl"
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl"

# Run training
cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction
python offline_train_2min.py
```

---

## What Each Variable Does

| Variable | Purpose | Example |
|----------|---------|---------|
| `DHAN_ACCESS_TOKEN` | Your Dhan API authentication token | `abc123xyz...` |
| `DHAN_CLIENT_ID` | Your Dhan client ID | `client_123` |
| `TRAIN_START_DATE` | Start date for training data | `2025-09-01` |
| `TRAIN_END_DATE` | End date for training data | `2025-12-07` |
| `XGB_PATH` | **Where to save XGB model** | `trained_models/production/xgb_model.pkl` |
| `NEUTRAL_PATH` | **Where to save neutrality model** | `trained_models/production/neutral_model.pkl` |

---

## Expected Output

When you run with all variables set correctly:

```
2025-12-07 14:30:00 | INFO | __main__ | Fetching 1-minute candles from 2025-09-01 to 2025-12-07
2025-12-07 14:35:15 | INFO | __main__ | Fetched 15000 candles
2025-12-07 14:40:00 | INFO | __main__ | Generating labels (BUY/SELL/FLAT)...
2025-12-07 14:45:30 | INFO | __main__ | Generated 12000 labels (BUY: 3500, SELL: 3800, FLAT: 4700)
2025-12-07 14:50:00 | INFO | __main__ | Training XGB directional model...
2025-12-07 14:55:00 | INFO | online_trainer | XGB training complete. AUC: 0.642
2025-12-07 14:57:00 | INFO | __main__ | Training neutrality model...
2025-12-07 14:58:30 | INFO | online_trainer | Neutrality training complete
2025-12-07 14:59:00 | INFO | __main__ | Offline 2-minute training complete.
```

---

## Verify Models Were Created

After training completes, check that files exist:

```bash
# Check XGB model
ls -lh trained_models/production/xgb_model.pkl
# Output: -rw-rw-r-- xgb_model.pkl (85 KB)

# Check neutrality model
ls -lh trained_models/production/neutral_model.pkl
# Output: -rw-rw-r-- neutral_model.pkl (35 KB)

# Check feature schema
ls -lh trained_models/production/feature_schema.json
# Output: -rw-rw-r-- feature_schema.json (3 KB)

# Check training data
ls -lh feature_log.csv
# Output: -rw-rw-r-- feature_log.csv (250 KB)
```

---

## Next Steps

1. **Set all 6 required environment variables** (see above)
2. **Run training command** (wait 5-15 minutes)
3. **Verify models were created** (check files exist)
4. **Restart automation** to use new models:
   ```bash
   sudo systemctl restart nifty-automation.service
   ```

---

## Documentation Files

- **TRAINING_QUICK_START.md** - Comprehensive training guide (JUST UPDATED)
- **DOCUMENTATION_INDEX.md** - Complete documentation index
- **backups/mdFiles/OPERATIONS_AND_TRAINING_GUIDE.md** - Archived detailed guide

---

## Quick Reference

| What You Want | Command |
|---|---|
| **Full training command** | See "Complete Working Command" above |
| **Just the variables** | See "Required Variables" above |
| **Check date format** | Use YYYY-MM-DD only (e.g., 2025-12-07) |
| **Verify models created** | `ls -lh trained_models/production/*.pkl` |
| **See training logs** | Watch terminal output while running |
| **Restart automation** | `sudo systemctl restart nifty-automation.service` |

---

## Summary

**The Key Fix:** Add these two environment variables to your training command:

```bash
export XGB_PATH="trained_models/production/xgb_model.pkl"
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl"
```

These tell the script WHERE to save the trained models. Without them, the script doesn't know where to write the output files.

**Full corrected command:** See "Complete Working Command" section above.

---

**Status:** ✅ FIXED - Ready to train  
**Documentation:** Updated TRAINING_QUICK_START.md with all corrections

