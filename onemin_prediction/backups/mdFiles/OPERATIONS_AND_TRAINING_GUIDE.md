# OPERATIONS & TRAINING GUIDE
**Project:** onemin_prediction – NIFTY Scalping Automation  
**Date:** December 7, 2025  
**Purpose:** How to run the automation and train models

---

## TABLE OF CONTENTS

1. [Running the Automation](#running-the-automation)
2. [Training the Models](#training-the-models)
3. [Trained Models Directory](#trained-models-directory)
4. [Label Generation & Data](#label-generation--data)
5. [Complete Training Workflow](#complete-training-workflow)
6. [Troubleshooting](#troubleshooting)

---

## RUNNING THE AUTOMATION

### Prerequisites

```bash
# 1. Install Python 3.10+
python3 --version  # Should be >= 3.10

# 2. Install required packages
pip install -r requirements.txt

# 3. Set environment variables
export DHAN_ACCESS_TOKEN="your_token_here"
export DHAN_CLIENT_ID="your_client_id"
export DHAN_HIST_BASE_URL="https://api.dhan.co"
export LOGLEVEL="INFO"
```

### Quick Start: Run at Market Open

**Option 1: Manual Startup (9:15 AM IST)**

```bash
# Navigate to project directory
cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction

# Run the main automation
python run_main.py
```

**Expected Output:**
```
2025-12-07 09:15:00 | INFO | main_event_loop | Connecting to Dhan API...
2025-12-07 09:15:05 | INFO | main_event_loop | WebSocket connected successfully
2025-12-07 09:15:10 | INFO | main_event_loop | Rule weights: IND=0.500 MTF=0.350 PAT=0.150
2025-12-07 09:15:10 | INFO | main_event_loop | Trade params: TP=0.150% SL=0.080%
2025-12-07 09:15:15 | INFO | main_event_loop | Global components initialized successfully
2025-12-07 09:15:20 | INFO | main_event_loop | NIFTY50 WebSocket connected
```

**Option 2: Automated with Cron (Recommended)**

Create a cron job to start automatically at 9:15 AM:

```bash
# Edit crontab
crontab -e

# Add this line (runs at 9:15 AM, Monday-Friday)
15 9 * * 1-5 cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction && python run_main.py >> logs/automation.log 2>&1
```

**Option 3: Systemd Service (Production)**

Create `/etc/systemd/system/nifty-automation.service`:

```ini
[Unit]
Description=NIFTY Scalping Automation
After=network.target

[Service]
Type=simple
User=hanumanth
WorkingDirectory=/home/hanumanth/Documents/sunflower-group_2/onemin_prediction
Environment="DHAN_ACCESS_TOKEN=your_token"
Environment="DHAN_CLIENT_ID=your_client"
Environment="LOGLEVEL=INFO"
ExecStart=/usr/bin/python3 run_main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable nifty-automation.service
sudo systemctl start nifty-automation.service
sudo systemctl status nifty-automation.service
```

### Configuration Options

**Key Environment Variables:**

```bash
# API Credentials (REQUIRED)
export DHAN_ACCESS_TOKEN="your_dhan_token"
export DHAN_CLIENT_ID="your_client_id"

# Trading Parameters
export TRADE_HORIZON_MIN=2           # Hold duration (minutes)
export TRADE_TP_PCT=0.0015           # Take profit (0.15%)
export TRADE_SL_PCT=0.0008           # Stop loss (0.08%)

# Gate Thresholds
export QMIN_BASE=0.12                # Margin threshold
export NEUTRAL_GATE=0.60             # Max neutral probability
export Q_PROB_GATE=0.55              # Q-model threshold

# Rule Weights (sum to 1.0)
export RULE_WEIGHT_IND=0.50          # Indicator weight
export RULE_WEIGHT_MTF=0.35          # Multi-TF weight
export RULE_WEIGHT_PAT=0.15          # Pattern weight

# Logging
export LOGLEVEL=INFO                 # INFO or DEBUG
export FEATURE_LOG=feature_log.csv   # Feature output file

# Feature Output
export INTRADAY_CACHE_ENABLE=1       # Cache 1-min candles
export INTRADAY_CACHE_DIR=data/intraday_cache/
```

### Monitoring the Live Automation

**1. Real-Time Signal Monitoring:**

```bash
# Watch signals as they're generated
tail -f trained_models/production/signals.jsonl

# Example output:
# {"timestamp":"2025-12-07T09:16:30","pred_for":"2025-12-07T09:18:30","direction":"BUY","price":18505,...}
# {"timestamp":"2025-12-07T09:17:30","pred_for":"2025-12-07T09:19:30","direction":"SELL","price":18490,...}
```

**2. Feature Logging (for Training):**

```bash
# Monitor features being logged for model retraining
tail -f feature_log.csv

# Columns: timestamp, label, buy_prob, alpha, tradeable, is_flat, tick_count, [features...]
```

**3. System Logs:**

```bash
# Check for errors and warnings
tail -f logs/main_event_loop.log

# Monitor memory usage
watch -n 5 'ps aux | grep python | grep run_main'
```

**4. Memory & Resources:**

```bash
# Monitor memory growth (should be stable after Fix #1)
while true; do
  free -h | grep Mem
  sleep 60
done
```

### Graceful Shutdown

**Stop the automation cleanly:**

```bash
# Method 1: Find process and terminate
ps aux | grep run_main
kill -15 <PID>  # Graceful shutdown (SIGTERM)

# Method 2: Using systemd
sudo systemctl stop nifty-automation.service

# Method 3: Keyboard interrupt
Ctrl+C  # In terminal where it's running

# Verify cleanup
ps aux | grep run_main  # Should show no running process
```

---

## TRAINING THE MODELS

### Model Training Workflow

```
┌─────────────────────────────────────────────────────┐
│  Market Close (3:30 PM)                             │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  Step 1: Collect Feature Data                       │
│  • feature_log.csv accumulated throughout day       │
│  • Contains: prices, features, probabilities        │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  Step 2: Generate Labels (Offline)                  │
│  • Run: offline_train_2min.py                       │
│  • Fetches historical candles from Dhan API         │
│  • Computes WIN/LOSS/NONE outcomes                  │
│  • Creates BUY/SELL/FLAT labels                     │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  Step 3: Train Models (Optional)                    │
│  • XGB directional model                            │
│  • Neutrality logistic classifier                   │
│  • Save to trained_models/production/               │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  Step 4: Validate Models (Optional)                 │
│  • Run: offline_eval_2min_full.py                   │
│  • Check AUC, accuracy, P&L                         │
│  • Compare to previous model                        │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  Step 5: Deploy (If Improved)                       │
│  • Backup old models                                │
│  • Copy new models to production/                   │
│  • Restart automation to use new models             │
└─────────────────────────────────────────────────────┘
```

### Step 1: Generate Labels (Required for Training)

Labels are generated from historical data. This step MUST be run offline.

**What Labels Are:**

```
Label Definition (2-minute horizon):
  
  Entry at close_t
  Horizon: t to t+2 minutes
  
  Long outcome:
    WIN  if high[t, t+2] >= close_t * (1 + TRADE_TP_PCT)
    LOSS if low[t, t+2]  <= close_t * (1 - TRADE_SL_PCT)
    NONE otherwise
  
  Short outcome:
    WIN  if low[t, t+2]  <= close_t * (1 - TRADE_TP_PCT)
    LOSS if high[t, t+2] >= close_t * (1 + TRADE_SL_PCT)
    NONE otherwise
  
  Final label:
    BUY  if long_outcome == "WIN" AND short_outcome != "WIN"
    SELL if short_outcome == "WIN" AND long_outcome != "WIN"
    FLAT otherwise
```

**Generate Labels:**

```bash
# Step 1: Generate labels for date range
python offline_train_2min.py \
  --start-date 2025-09-01 \
  --end-date 2025-12-07 \
  --output labeled_data.csv

# Expected output:
# 2025-12-07 16:00:00 | INFO | offline_train_2min | Fetching candles: 2025-12-01 to 2025-12-07
# 2025-12-07 16:05:00 | INFO | offline_train_2min | Fetched 1440 candles (7 days × ~200 candles/day)
# 2025-12-07 16:10:00 | INFO | offline_train_2min | Generated 1200 labels (BUY: 350, SELL: 380, FLAT: 470)
# 2025-12-07 16:15:00 | INFO | offline_train_2min | Saved to feature_log.csv
```

**Verify Labels Were Created:**

```bash
# Check feature_log.csv
head -5 feature_log.csv

# Example output (columns):
# timestamp,user,label,buy_prob,alpha,tradeable,is_flat,tick_count,ema_8,ema_21,ema_50,ta_rsi14,...
# 2025-12-07T16:00:30,USER,BUY,0.720000,0.0,True,False,123,18510,18490,18480,55,...
# 2025-12-07T16:01:30,USER,SELL,0.280000,0.0,True,False,115,18505,18495,18485,42,...
# 2025-12-07T16:02:30,USER,FLAT,0.510000,0.0,True,True,98,18500,18500,18490,50,...

# Count labels
grep -c "^" feature_log.csv  # Total rows
grep -c ",BUY," feature_log.csv  # BUY labels
grep -c ",SELL," feature_log.csv  # SELL labels
grep -c ",FLAT," feature_log.csv  # FLAT labels
```

### Step 2: Train XGB Model (Optional)

Once you have labeled data, train the directional model:

```bash
# Train XGB model on labeled data
python offline_train_2min.py \
  --mode train \
  --input feature_log.csv \
  --output-model trained_models/production/xgb_model.pkl

# Expected output:
# 2025-12-07 16:30:00 | INFO | online_trainer | Training XGB on 1200 labeled rows
# 2025-12-07 16:30:00 | INFO | online_trainer | Non-FLAT rows: 730 (BUY: 350, SELL: 380)
# 2025-12-07 16:31:00 | INFO | online_trainer | XGB training complete
# 2025-12-07 16:31:05 | INFO | online_trainer | Model saved to trained_models/production/xgb_model.pkl
# 2025-12-07 16:31:10 | INFO | online_trainer | Training XGB accuracy: 0.64 (on training set)
```

### Step 3: Train Neutrality Model (Optional)

Neutrality model predicts if market is too choppy to trade:

```bash
# Train neutrality classifier
python offline_train_2min.py \
  --mode train-neutral \
  --input feature_log.csv \
  --output-model trained_models/production/neutral_model.pkl

# Expected output:
# 2025-12-07 16:35:00 | INFO | online_trainer | Training Neutrality on 1200 rows
# 2025-12-07 16:35:05 | INFO | online_trainer | FLAT rows: 470 (39%), non-FLAT: 730 (61%)
# 2025-12-07 16:36:00 | INFO | online_trainer | Neutrality training complete
# 2025-12-07 16:36:05 | INFO | online_trainer | Model saved to trained_models/production/neutral_model.pkl
```

### Step 4: Evaluate Models (Validation)

Evaluate model performance before deployment:

```bash
# Evaluate models on test set
python offline_eval_2min_full.py \
  --input feature_log.csv \
  --model-path trained_models/production/xgb_model.pkl \
  --output-report evaluation.txt

# Expected output:
# 2025-12-07 16:40:00 | INFO | offline_eval | Loading model...
# 2025-12-07 16:40:05 | INFO | offline_eval | Evaluating on 300 test samples (25% holdout)
# 2025-12-07 16:40:10 | INFO | offline_eval | 
# ═══════════════════════════════════════════════════
# MODEL EVALUATION RESULTS
# ═══════════════════════════════════════════════════
# 
# AUC-ROC:               0.642
# Accuracy:              0.638
# Precision (BUY):       0.615
# Recall (BUY):          0.580
# 
# Wins on tradeable:     62% (186 / 300)
# Avg winner:            +25 pips
# Avg loser:             -12 pips
# Expectancy:            +13 pips/trade
# Sharpe (annualized):   1.24
# 
# ═══════════════════════════════════════════════════
```

---

## TRAINED MODELS DIRECTORY

### Directory Structure

```
trained_models/
├── production/                    # Live models (used by automation)
│   ├── xgb_model.pkl             # XGB directional model (may not exist yet)
│   ├── neutral_model.pkl         # Neutrality classifier (may not exist yet)
│   ├── q_model_2min.json         # Q-model (pre-trained, read-only)
│   ├── feature_schema.json       # Feature column names (auto-generated)
│   ├── fut_candles_vwap_cvd.csv  # Futures VWAP/CVD cache (reference data)
│   ├── fut_ticks_vwap_cvd.csv    # Futures tick data (reference data)
│   └── signals.jsonl             # Live signals (generated by automation)
│
└── experiments/                   # Experimental/backup models
    └── feature_schema.json       # Schema from experiments
```

### What Each File Does

**xgb_model.pkl** (XGB Directional Model)
```
Purpose: Predicts direction (BUY vs SELL)
Input:   50+ features
Output:  p_buy (probability of buying), p_sell = 1 - p_buy
Status:  ⚠️ May not exist initially (created after first training)
Train:   python offline_train_2min.py --mode train
Size:    ~50-100 KB
Update:  Weekly or when P&L improves
```

**neutral_model.pkl** (Neutrality Classifier)
```
Purpose: Detects choppy markets
Input:   Same 50+ features
Output:  p_flat (probability that market is neutral/choppy)
Status:  ⚠️ May not exist initially (created after first training)
Train:   python offline_train_2min.py --mode train-neutral
Size:    ~30 KB
Update:  Weekly or when chop detection needs tuning
```

**q_model_2min.json** (Q-Model Confidence Estimator)
```
Purpose: Estimates confidence in directional prediction
Input:   Direction (BUY/SELL) + features
Output:  Q_score (0 to 1, confidence level)
Status:  ✅ Pre-trained, ready to use
Created: During initial setup
Size:    ~5 KB
Update:  Not needed for basic operation
```

**feature_schema.json** (Feature Column Mapping)
```
Purpose: Maps feature names to column indices
Example:
  {
    "ema_8": 0,
    "ema_21": 1,
    "ema_50": 2,
    "ta_rsi14": 3,
    ...
  }
Status:  ✅ Auto-generated during training
Created: When XGB model is trained
Size:    ~3 KB
Update:  Auto-updated with each training
```

**signals.jsonl** (Live Signals Log)
```
Purpose: Record of all signals generated
Format:  One JSON object per line
Example: {"timestamp":"2025-12-07T09:16:30","direction":"BUY","price":18505,"buy_prob":0.68,...}
Status:  ✅ Auto-created, appends every minute
Created: First time automation runs
Size:    ~1-5 KB per day
Update:  Real-time (append-only)
```

**fut_candles_vwap_cvd.csv** (Futures Reference Data)
```
Purpose: Futures VWAP and CVD for sidecar features
Format:  Timestamp, VWAP, CVD, Volume
Status:  Optional (for advanced features)
Created: When futures sidecar is enabled
Size:    ~100 KB
Update:  Daily or as needed
```

### Current Status Check

```bash
# Verify what models exist
ls -lh trained_models/production/

# Expected output (before any training):
# -rw-rw-r--  q_model_2min.json (5 KB)           ✅ Ready
# -rw-rw-r--  feature_schema.json (3 KB)         ✅ Ready
# -rw-rw-r--  fut_candles_vwap_cvd.csv (80 KB)   ✅ Optional
# -rw-rw-r--  fut_ticks_vwap_cvd.csv (200 KB)    ✅ Optional
# -rw-rw-r--  signals.jsonl (2 KB)               ✅ Created on first run

# After training:
# -rw-rw-r--  xgb_model.pkl (85 KB)              ✅ New (after training)
# -rw-rw-r--  neutral_model.pkl (35 KB)          ✅ New (after training)
```

---

## LABEL GENERATION & DATA

### What Data is Needed for Training?

**Required:**
1. ✅ Historical 1-minute OHLC candles (fetched from Dhan API)
2. ✅ Trade parameters (TP%, SL%, horizon minutes)
3. ✅ Current prices and indicators

**Generated During Live Trading:**
1. ✅ Feature vectors (50+ indicators) → saved in feature_log.csv
2. ✅ Model predictions (buy_prob, neutral_prob)
3. ✅ Trade outcomes (WIN/LOSS/NONE) → labels in feature_log.csv

### Label Generation Process (Detailed)

**Minute t (Signal Time):**
```
Price: 18,505
Automation generates signal:
  → XGB prediction: p_buy = 0.68
  → Buy probability recorded
  → Features recorded
  → Record written to feature_log.csv (with label = FLAT initially)
```

**Minute t+2 (Outcome Time):**
```
Automation checks outcome:
  Entry price: 18,505
  TP: 18,505 * 1.0015 = 18,532.77
  SL: 18,505 * 0.9992 = 18,499.04
  
  Actual high[t, t+2]: 18,535
  Actual low[t, t+2]: 18,498
  
  Analysis:
    HIGH >= TP? YES (18,535 >= 18,533) → BUY wins
    LOW <= SL? YES (18,498 <= 18,499) → SELL wins
    
  BUT: BUY hit TP first (in same minute)
  → Label = BUY
  
Feature log entry UPDATED:
  Old: timestamp, USER, FLAT, 0.68, ..., [features]
  New: timestamp, USER, BUY, 0.68, ..., [features]
```

### Verify Labels Are Created

**Check #1: Feature log file exists**
```bash
ls -lh feature_log.csv
# Output: -rw-rw-r-- feature_log.csv (50 KB)  ✅ Exists
```

**Check #2: Labels are non-FLAT**
```bash
# Count different labels
tail -100 feature_log.csv | grep -c ",BUY,"
tail -100 feature_log.csv | grep -c ",SELL,"
tail -100 feature_log.csv | grep -c ",FLAT,"

# Should see mostly FLAT at first (trading not happening)
# After several days of trading, should see ~60% non-FLAT
```

**Check #3: Verify label quality**
```bash
# Verify labels are not all the same
sort feature_log.csv | uniq -c | head -5

# Should see variety of BUY/SELL/FLAT
```

---

## COMPLETE TRAINING WORKFLOW

### Recommended Weekly Training Schedule

**Monday-Friday (During market hours):**
```
9:15 AM - 3:30 PM:  Automation runs, generates signals
                     Features logged to feature_log.csv
                     Labels generated for previous day's signals
```

**Friday Evening (After market close):**
```
4:00 PM:  Compile all week's feature data
          python offline_train_2min.py --mode generate-features

4:15 PM:  Train directional model
          python offline_train_2min.py --mode train

4:30 PM:  Train neutrality model
          python offline_train_2min.py --mode train-neutral

5:00 PM:  Evaluate new models
          python offline_eval_2min_full.py

5:30 PM:  Review results
          Compare new AUC vs old AUC
          If AUC improved ≥ 2%, deploy new model
          Otherwise, keep existing model
```

**Saturday:**
```
Backtest new models on historical data
python offline_eval_2min_full.py --backtest-window 30d
```

**Sunday Evening:**
```
Deploy if validation passed
Restart automation on Monday with new models
```

### Complete Training Script

**train.sh** (Automated training script):

```bash
#!/bin/bash

# Complete training workflow

set -e  # Exit on error

echo "════════════════════════════════════════"
echo "  NIFTY AUTOMATION - WEEKLY TRAINING"
echo "════════════════════════════════════════"

# Configuration
PROJECT_DIR="/home/hanumanth/Documents/sunflower-group_2/onemin_prediction"
BACKUP_DIR="$PROJECT_DIR/trained_models/backups/$(date +%Y%m%d)"
OUTPUT_REPORT="$PROJECT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "[1/5] Backing up current models..."
mkdir -p "$BACKUP_DIR"
cp "$PROJECT_DIR/trained_models/production/xgb_model.pkl" "$BACKUP_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/trained_models/production/neutral_model.pkl" "$BACKUP_DIR/" 2>/dev/null || true
echo "✅ Backup saved to: $BACKUP_DIR"

echo ""
echo "[2/5] Generating labels..."
cd "$PROJECT_DIR"
python offline_train_2min.py --start-date $(date -d "7 days ago" +%Y-%m-%d) --end-date $(date +%Y-%m-%d)
echo "✅ Labels generated"

echo ""
echo "[3/5] Training XGB directional model..."
python offline_train_2min.py --mode train --input feature_log.csv
echo "✅ XGB model trained"

echo ""
echo "[4/5] Training neutrality model..."
python offline_train_2min.py --mode train-neutral --input feature_log.csv
echo "✅ Neutrality model trained"

echo ""
echo "[5/5] Evaluating models..."
python offline_eval_2min_full.py --input feature_log.csv | tee "$OUTPUT_REPORT"

echo ""
echo "════════════════════════════════════════"
echo "  TRAINING COMPLETE"
echo "════════════════════════════════════════"
echo "Report saved to: $OUTPUT_REPORT"
echo "Models available at: trained_models/production/"
echo ""
echo "Next: Review report, then restart automation:"
echo "  sudo systemctl restart nifty-automation.service"
```

Run the training script:
```bash
chmod +x train.sh
./train.sh
```

---

## TROUBLESHOOTING

### Problem 1: No Labels Being Generated

**Symptom:**
```
feature_log.csv has only FLAT labels, no BUY/SELL
```

**Cause:**
- Automation hasn't generated enough signals yet
- Labels are created with 2-minute delay (at t+2)

**Solution:**
```bash
# Wait for at least 1 full day of trading
# Then check:
tail -50 feature_log.csv | grep -E "BUY|SELL"

# If still only FLAT after 2 days:
# Check if signals are being generated:
tail -20 trained_models/production/signals.jsonl

# If no signals:
# Check logs:
tail -100 logs/main_event_loop.log | grep "SIGNAL"
```

### Problem 2: Training Fails (Insufficient Data)

**Error:**
```
ValueError: Not enough samples to train
(minimum 220 rows required, got 50)
```

**Solution:**
```bash
# Need at least 220 labeled rows (non-FLAT)
# At 8-12 trades/day, takes 2-3 weeks
# Meanwhile, use pre-trained models

# Verify minimum data requirement
wc -l feature_log.csv
grep -c ",BUY\|,SELL" feature_log.csv  # Count non-FLAT rows
```

### Problem 3: Model Performance Degraded

**Symptom:**
```
AUC dropped from 0.64 to 0.52 after retraining
Win rate dropped from 62% to 48%
```

**Cause:**
- Market conditions changed (trending to choppy, etc.)
- Too few new samples
- Model overfitting

**Solution:**
```bash
# Don't deploy new model if:
# 1. AUC dropped more than 5%
# 2. Test samples < 200

# Stick with previous model:
cp trained_models/backups/20251201/xgb_model.pkl \
   trained_models/production/xgb_model.pkl

# Restart automation with old model
sudo systemctl restart nifty-automation.service
```

### Problem 4: Automation Crashes During Training

**Error:**
```
RuntimeError: Cannot pickle model while training
```

**Solution:**
```bash
# Training and production cannot run simultaneously
# Stop automation before training:
sudo systemctl stop nifty-automation.service

# Run training
./train.sh

# Restart automation
sudo systemctl start nifty-automation.service
```

### Problem 5: Feature Schema Mismatch

**Error:**
```
ValueError: Feature count mismatch (50 vs 48)
```

**Cause:**
- Model was trained with different features
- Feature_schema.json is outdated

**Solution:**
```bash
# Update feature schema
python offline_train_2min.py --mode update-schema

# Or manually align:
# Copy production schema to experiments
cp trained_models/production/feature_schema.json \
   trained_models/experiments/feature_schema.json
```

---

## QUICK REFERENCE

### Starting the Automation

```bash
# Manual start (testing)
python run_main.py

# Via systemd (production)
sudo systemctl start nifty-automation.service

# Check status
sudo systemctl status nifty-automation.service

# Stop gracefully
sudo systemctl stop nifty-automation.service
```

### Training the Models

```bash
# Generate labels (offline)
python offline_train_2min.py --start-date 2025-12-01 --end-date 2025-12-07

# Train XGB model
python offline_train_2min.py --mode train --input feature_log.csv

# Train neutrality model
python offline_train_2min.py --mode train-neutral --input feature_log.csv

# Evaluate models
python offline_eval_2min_full.py --input feature_log.csv
```

### Monitoring

```bash
# Live signals
tail -f trained_models/production/signals.jsonl

# Feature logs (training data)
tail -f feature_log.csv

# System logs
tail -f logs/main_event_loop.log

# Check memory usage
watch -n 5 'ps aux | grep python'
```

### File Locations

```
Automation:        run_main.py
Main event loop:   main_event_loop.py
Training scripts:  offline_train_2min.py
Evaluation:        offline_eval_2min_full.py
Models:            trained_models/production/
Signals:           trained_models/production/signals.jsonl
Feature data:      feature_log.csv
Logs:              logs/main_event_loop.log
Configuration:     run_main.py (SimpleNamespace config)
```

---

**Created:** December 7, 2025  
**Status:** Complete  
**Ready for:** Live trading and model training

