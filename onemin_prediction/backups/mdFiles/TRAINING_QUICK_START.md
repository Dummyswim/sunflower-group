# TRAINING QUICK START GUIDE

**Purpose:** How to correctly run offline_train_2min.py  
**Last Updated:** December 7, 2025
**Note:** This guide now covers **cache-aware training** which reduces API calls by 80-90%

---

## ⚡ NEW: Cache-Aware Training

The training system now automatically:
- ✅ Loads cached data from disk (fast, instant)
- ✅ Fetches only missing dates from API (reduces API calls)
- ✅ Caches newly fetched data for future runs (6-8x speedup on reruns)

**Result:** First run takes ~30 min, subsequent runs take ~2 min!

See `CACHE_AWARE_TRAINING.md` for detailed cache configuration.

---

## ❌ WHAT DOESN'T WORK

```bash
# ❌ Wrong - Script doesn't accept command-line arguments
python offline_train_2min.py \
  --start-date 2025-09-01 \
  --end-date 2025-12-07 \
  --output labeled_data.csv

ERROR: TRAIN_START_DATE and TRAIN_END_DATE must be set
```

```bash
# ❌ Incomplete - Missing model output paths
export TRAIN_START_DATE="2025-09-01"
export TRAIN_END_DATE="2025-12-07"
python offline_train_2min.py

ERROR: XGB_PATH and NEUTRAL_PATH must be set for offline training outputs
```

---

## ✅ WHAT WORKS - USE ENVIRONMENT VARIABLES

### Quick Start (Copy & Paste) - WITH Cache

```bash
# Set your Dhan API credentials
export DHAN_ACCESS_TOKEN="your_actual_token_here"
export DHAN_CLIENT_ID="your_actual_client_id_here"

# Set training date range
export TRAIN_START_DATE="2025-09-01"
export TRAIN_END_DATE="2025-12-07"

# Set model output paths (REQUIRED!)
export XGB_PATH="trained_models/production/xgb_model.pkl"
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl"

# Enable cache (automatic if not set, but good to be explicit)
export INTRADAY_CACHE_ENABLE="1"
export INTRADAY_CACHE_DIR="data/intraday_cache"

# Run training
cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction
python offline_train_2min.py
```

**What happens:**
```
=== CACHE STATUS ===
Cache directory: data/intraday_cache
Total cached dates: 743
Date range: 2025-09-01 to 2025-12-07
Trading days in range: 65
Cached: 60, Missing: 5
First missing dates: 2025-12-06, 2025-12-07
===================
INFO | Loaded 450000 candles from cache
INFO | Fetching 5 missing dates from API
INFO | Total candles in range: 195000
INFO | Offline 2-minute training complete.
```

---

## Complete Command (All Options)

```bash
# Set API credentials
export DHAN_ACCESS_TOKEN="your_token"
export DHAN_CLIENT_ID="your_client_id"

# Set training dates (REQUIRED)
export TRAIN_START_DATE="2025-09-01"           # Start date (YYYY-MM-DD)
export TRAIN_END_DATE="2025-12-07"             # End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)

# Set model output paths (REQUIRED)
export XGB_PATH="trained_models/production/xgb_model.pkl"          # XGB model output
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl"  # Neutrality model output

# Set trading parameters (optional - uses defaults if not set)
export TRADE_HORIZON_MIN=2                     # Hold duration in minutes (default: 2)
export TRADE_TP_PCT=0.0015                     # Take profit % (default: 0.0015 = 0.15%)
export TRADE_SL_PCT=0.0008                     # Stop loss % (default: 0.0008 = 0.08%)

# Enable caching (optional)
export INTRADAY_CACHE_ENABLE=1                 # Cache 1-min candles (default: 1)
export INTRADAY_CACHE_DIR="data/intraday_cache" # Cache directory

# Run training
python offline_train_2min.py
```

---

## Environment Variables Explained

| Variable | Type | Format | Example | Notes |
|----------|------|--------|---------|-------|
| `DHAN_ACCESS_TOKEN` | Required | String | `abc123...` | Your Dhan API token |
| `DHAN_CLIENT_ID` | Required | String | `client_123` | Your Dhan client ID |
| `TRAIN_START_DATE` | Required | Date | `2025-09-01` | Start date YYYY-MM-DD |
| `TRAIN_END_DATE` | Required | Date | `2025-12-07` | End date YYYY-MM-DD |
| `XGB_PATH` | Required | Path | `trained_models/production/xgb_model.pkl` | Where to save XGB model |
| `NEUTRAL_PATH` | Required | Path | `trained_models/production/neutral_model.pkl` | Where to save neutrality model |
| `TRADE_HORIZON_MIN` | Optional | Integer | `2` | Trade hold duration (minutes) |
| `TRADE_TP_PCT` | Optional | Float | `0.0015` | Take profit (as decimal, 0.0015 = 0.15%) |
| `TRADE_SL_PCT` | Optional | Float | `0.0008` | Stop loss (as decimal, 0.0008 = 0.08%) |
| `INTRADAY_CACHE_ENABLE` | Optional | Integer | `1` | Enable caching (1=yes, 0=no) |
| `INTRADAY_CACHE_DIR` | Optional | Path | `data/intraday_cache` | Cache directory path |

---

## Date Format

**Accepted formats:**

```
✅ YYYY-MM-DD
   export TRAIN_START_DATE="2025-09-01"
   export TRAIN_END_DATE="2025-12-07"

✅ YYYY-MM-DD HH:MM:SS
   export TRAIN_START_DATE="2025-09-01 09:30:00"
   export TRAIN_END_DATE="2025-12-07 15:30:00"

❌ Other formats (will fail)
   --start-date 2025-09-01           (command-line arg)
   09/01/2025                        (MM/DD/YYYY)
   01-Sep-2025                       (text format)
```

---

## Step-by-Step Procedure

### Step 1: Set Credentials

```bash
export DHAN_ACCESS_TOKEN="your_dhan_token_here"
export DHAN_CLIENT_ID="your_client_id_here"

# Verify they're set
echo $DHAN_ACCESS_TOKEN
echo $DHAN_CLIENT_ID
```

### Step 2: Navigate to Project Directory

```bash
cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction
```

### Step 3: Set Training Dates

```bash
# For historical data (September to December 2025)
export TRAIN_START_DATE="2025-09-01"
export TRAIN_END_DATE="2025-12-07"

# Or for last 30 days
export TRAIN_START_DATE="2025-11-07"
export TRAIN_END_DATE="2025-12-07"

# Or specific date range
export TRAIN_START_DATE="2025-10-15"
export TRAIN_END_DATE="2025-12-01"
```

### Step 4: Set Model Output Paths

```bash
# Where to save the trained models
export XGB_PATH="trained_models/production/xgb_model.pkl"
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl"
```

### Step 5: Run Training

```bash
python offline_train_2min.py
```

### Step 6: Monitor Output

You should see:

```
2025-12-07 02:00:30 | INFO | __main__ | Fetching 1-minute candles...
2025-12-07 02:00:35 | INFO | __main__ | Fetched 15000 candles (100 days)
2025-12-07 02:01:00 | INFO | __main__ | Generating labels...
2025-12-07 02:02:00 | INFO | __main__ | Generated 12000 labels (BUY: 3500, SELL: 3800, FLAT: 4700)
2025-12-07 02:02:15 | INFO | __main__ | Training XGB model...
2025-12-07 02:03:00 | INFO | __main__ | Model trained. AUC: 0.642
2025-12-07 02:03:30 | INFO | __main__ | Training complete!
```

---

## What Happens

1. **Fetches candles:** Retrieves 1-minute OHLC from Dhan API for date range
2. **Generates labels:** For each bar, simulates trade and checks outcome (BUY/SELL/FLAT)
3. **Builds features:** Creates 50+ technical indicators
4. **Trains models:** 
   - XGBoost directional model (BUY vs SELL)
   - Neutrality logistic classifier (FLAT detection)
5. **Saves output:**
   - Models: `trained_models/production/xgb_model.pkl`, `neutral_model.pkl`
   - Features: `trained_models/production/feature_schema.json`
   - Labels: `feature_log.csv`

---

## Output Files

After training completes, you'll have:

```
trained_models/production/
├── xgb_model.pkl              ← XGB directional model (new/updated)
├── neutral_model.pkl          ← Neutrality classifier (new/updated)
├── feature_schema.json        ← Feature mapping (updated)
└── [others preserved]

Feature logs:
└── feature_log.csv            ← Training data with labels
```

---

## Troubleshooting

### Error: "XGB_PATH and NEUTRAL_PATH must be set"

**Cause:** Model output paths not configured

**Fix:**
```bash
export XGB_PATH="trained_models/production/xgb_model.pkl"
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl"
python offline_train_2min.py
```

### Error: "TRAIN_START_DATE and TRAIN_END_DATE must be set"

**Cause:** Environment variables not set

**Fix:**
```bash
export TRAIN_START_DATE="2025-09-01"
export TRAIN_END_DATE="2025-12-07"
python offline_train_2min.py
```

### Error: "DHAN_ACCESS_TOKEN is not set"

**Cause:** Dhan credentials not configured

**Fix:**
```bash
export DHAN_ACCESS_TOKEN="your_token"
export DHAN_CLIENT_ID="your_client_id"
python offline_train_2min.py
```

### Error: "Invalid TRAIN_START_DATE / TRAIN_END_DATE"

**Cause:** Date format incorrect

**Fix:** Use YYYY-MM-DD format only:
```bash
export TRAIN_START_DATE="2025-09-01"     # ✅ Correct
export TRAIN_START_DATE="09-01-2025"     # ❌ Wrong
export TRAIN_START_DATE="Sep 1, 2025"    # ❌ Wrong
```

### Error: "TRAIN_END_DATE must be >= TRAIN_START_DATE"

**Cause:** End date before start date

**Fix:** Ensure end date is after or equal to start date:
```bash
export TRAIN_START_DATE="2025-09-01"
export TRAIN_END_DATE="2025-12-07"       # Dec 7 is after Sep 1 ✅
```

### Training is slow / hangs

**Cause:** Large date range, API rate limiting, or network issues

**Solutions:**
1. Try a smaller date range first: `2025-12-01` to `2025-12-07`
2. Check internet connection
3. Verify Dhan API status
4. Enable caching: `export INTRADAY_CACHE_ENABLE=1`

### Models not updating

**Cause:** Models already exist, or insufficient data

**Solutions:**
1. Ensure training completed successfully
2. Check if 200+ labeled rows generated (minimum for training)
3. Verify `feature_log.csv` has BUY/SELL labels

---

## Complete Working Example

```bash
#!/bin/bash
# Training script - save as train.sh

set -e

# Configuration
PROJECT_DIR="/home/hanumanth/Documents/sunflower-group_2/onemin_prediction"
DHAN_TOKEN="your_token_here"
DHAN_CLIENT="your_client_id_here"

# Navigate to project
cd "$PROJECT_DIR"

# Set credentials
export DHAN_ACCESS_TOKEN="$DHAN_TOKEN"
export DHAN_CLIENT_ID="$DHAN_CLIENT"

# Set training dates (last 100 days)
export TRAIN_START_DATE="2025-09-01"
export TRAIN_END_DATE="2025-12-07"

# Set model output paths (REQUIRED)
export XGB_PATH="trained_models/production/xgb_model.pkl"
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl"

# Set trading parameters
export TRADE_HORIZON_MIN=2
export TRADE_TP_PCT=0.0015
export TRADE_SL_PCT=0.0008

# Enable caching
export INTRADAY_CACHE_ENABLE=1

# Run training
echo "Starting training..."
python offline_train_2min.py

echo "Training complete!"
echo "Models saved to:"
echo "  - $XGB_PATH"
echo "  - $NEUTRAL_PATH"
echo "Labels saved to: feature_log.csv"
```

**Run it:**
```bash
chmod +x train.sh
./train.sh
```

---

## Related Commands

### Train and Evaluate

```bash
# Train models
python offline_train_2min.py

# Evaluate models
export FEATURE_LOG="feature_log.csv"
python offline_eval_2min_full.py
```

### Check Training Data

```bash
# View feature_log.csv (first 5 rows)
head -5 feature_log.csv

# Count labels
grep -c ",BUY," feature_log.csv
grep -c ",SELL," feature_log.csv
grep -c ",FLAT," feature_log.csv

# Get statistics
wc -l feature_log.csv
```

### Verify Models Saved

```bash
ls -lh trained_models/production/xgb_model.pkl
ls -lh trained_models/production/neutral_model.pkl
```

---

## Key Points

✅ **Use environment variables**, not command-line arguments  
✅ **Set TRAIN_START_DATE and TRAIN_END_DATE** (both required)  
✅ **Set XGB_PATH and NEUTRAL_PATH** (both required for model output)  
✅ **Use YYYY-MM-DD format** for dates  
✅ **Set DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID** first  
✅ **Training takes 5-15 minutes** depending on date range  
✅ **Models save automatically** to XGB_PATH and NEUTRAL_PATH  
✅ **Check feature_log.csv** for labels and training data  

---

**Need help?** Check DOCUMENTATION_INDEX.md or OPERATIONS_AND_TRAINING_GUIDE.md in backups/mdFiles/

