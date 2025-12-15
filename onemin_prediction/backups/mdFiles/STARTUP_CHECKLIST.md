# STARTUP CHECKLIST - MARKET OPEN PROCEDURE

**Purpose:** Verify system is ready before market opens at 9:15 AM IST

---

## PRE-MARKET CHECKLIST (9:00 AM - 9:14 AM)

### ✅ Check 1: Environment Setup

```bash
# Verify environment variables are set
echo "DHAN_ACCESS_TOKEN: $DHAN_ACCESS_TOKEN"
echo "DHAN_CLIENT_ID: $DHAN_CLIENT_ID"
echo "LOGLEVEL: $LOGLEVEL"

# If any variable is empty, export them:
export DHAN_ACCESS_TOKEN="your_token_here"
export DHAN_CLIENT_ID="your_client_id"
export LOGLEVEL="INFO"
```

**Result:** All environment variables populated ✅

---

### ✅ Check 2: Python Environment

```bash
# Verify Python version
python3 --version  # Should be 3.10+

# Verify required packages
python3 -c "import pandas, xgboost, dhan, websocket; print('✅ All packages installed')"
```

**Result:** Python 3.10+ and all packages available ✅

---

### ✅ Check 3: Directory Structure

```bash
# Navigate to project
cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction

# Verify directory structure
test -d trained_models/production && echo "✅ trained_models/production exists"
test -f trained_models/production/feature_schema.json && echo "✅ feature_schema.json exists"
test -f trained_models/production/q_model_2min.json && echo "✅ q_model_2min.json exists"
test -d logs && echo "✅ logs directory exists"
test -d data/intraday_cache && echo "✅ data/intraday_cache exists"
```

**Result:** All directories and required files present ✅

---

### ✅ Check 4: Required Files

```bash
# Verify all Python scripts present
test -f run_main.py && echo "✅ run_main.py"
test -f main_event_loop.py && echo "✅ main_event_loop.py"
test -f core_handler.py && echo "✅ core_handler.py"
test -f feature_pipeline.py && echo "✅ feature_pipeline.py"
test -f model_pipeline.py && echo "✅ model_pipeline.py"
test -f online_trainer.py && echo "✅ online_trainer.py"
test -f logging_setup.py && echo "✅ logging_setup.py"
```

**Result:** All required Python scripts present ✅

---

### ✅ Check 5: Dhan API Connectivity (CRITICAL)

```bash
# Test API connectivity
python3 << 'EOF'
import os
import requests
from base64 import b64encode

# Get credentials
token = os.getenv('DHAN_ACCESS_TOKEN', '')
client_id = os.getenv('DHAN_CLIENT_ID', '')

if not token or not client_id:
    print("❌ FAIL: Credentials not set")
    exit(1)

# Test API call
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

try:
    # Test endpoint: Get account info
    response = requests.get(
        'https://api.dhan.co/v2/accounts',
        headers=headers,
        timeout=5
    )
    
    if response.status_code == 200:
        print("✅ Dhan API connectivity: OK")
        print(f"   Account status: {response.status_code}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Connection Error: {e}")
    exit(1)
EOF
```

**Result:** API connectivity verified ✅ OR ❌ **STOP - Fix API issues before proceeding**

---

### ✅ Check 6: Syntax Validation

```bash
# Check Python syntax
python3 -m py_compile run_main.py && echo "✅ run_main.py syntax OK"
python3 -m py_compile main_event_loop.py && echo "✅ main_event_loop.py syntax OK"
python3 -m py_compile core_handler.py && echo "✅ core_handler.py syntax OK"
python3 -m py_compile feature_pipeline.py && echo "✅ feature_pipeline.py syntax OK"
python3 -m py_compile model_pipeline.py && echo "✅ model_pipeline.py syntax OK"
```

**Result:** All Python files valid syntax ✅

---

### ✅ Check 7: Permissions

```bash
# Verify write permissions
test -w . && echo "✅ Current directory writable"
test -w logs/ && echo "✅ logs/ directory writable"
test -w data/intraday_cache/ && echo "✅ data/intraday_cache/ writable"
test -w trained_models/production/ && echo "✅ trained_models/production/ writable"
```

**Result:** All write permissions correct ✅

---

### ✅ Check 8: No Existing Process

```bash
# Verify automation not already running
if pgrep -f "python.*run_main" > /dev/null; then
    echo "❌ FAIL: Automation already running"
    echo "   Kill existing process:"
    pkill -f "python.*run_main"
else
    echo "✅ No existing automation process"
fi
```

**Result:** Clean process state ✅

---

## STARTUP PROCEDURE (9:15 AM)

### Option A: Manual Startup

```bash
# Navigate to project
cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction

# Start automation
python run_main.py
```

**Expected Output (within 10 seconds):**
```
2025-12-07 09:15:00 | INFO | main_event_loop | ════════════════════════════════════
2025-12-07 09:15:00 | INFO | main_event_loop | NIFTY SCALPING AUTOMATION STARTED
2025-12-07 09:15:00 | INFO | main_event_loop | ════════════════════════════════════
2025-12-07 09:15:02 | INFO | main_event_loop | Connecting to Dhan WebSocket...
2025-12-07 09:15:05 | INFO | main_event_loop | WebSocket connected successfully
2025-12-07 09:15:05 | INFO | main_event_loop | Subscribing to NIFTY50 ticks...
2025-12-07 09:15:10 | INFO | main_event_loop | Global components initialized
2025-12-07 09:15:15 | INFO | main_event_loop | Rule weights: IND=0.500 MTF=0.350 PAT=0.150
2025-12-07 09:15:15 | INFO | main_event_loop | Trade params: TP=0.150% SL=0.080%
2025-12-07 09:15:20 | INFO | main_event_loop | Ready for signals. Waiting for market data...
```

**✅ Success:** Automation running, ready for trades

---

### Option B: Systemd Service Start

```bash
# Start via systemd
sudo systemctl start nifty-automation.service

# Verify running
sudo systemctl status nifty-automation.service

# Expected output:
# ● nifty-automation.service - NIFTY Scalping Automation
#      Loaded: loaded
#      Active: active (running)
#      Main PID: 12345
```

**✅ Success:** Service started

---

### Option C: Cron Automated Start

```bash
# Edit crontab
crontab -e

# Add line for 9:15 AM start (Monday-Friday)
15 9 * * 1-5 cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction && python run_main.py >> logs/automation.log 2>&1

# Verify cron entry
crontab -l
```

**✅ Success:** Cron configured, will auto-start at 9:15 AM

---

## MONITORING (During Market Hours)

### Monitor 1: Real-Time Signals

```bash
# Watch signals in real-time
tail -f trained_models/production/signals.jsonl

# Expected output every 1-2 minutes during active setup:
# {"timestamp":"2025-12-07T09:30:00","direction":"BUY","price":18505,"buy_prob":0.68,"setup_type":"MTF+IND"}
# {"timestamp":"2025-12-07T09:31:30","direction":"SELL","price":18490,"buy_prob":0.35,"setup_type":"PAT"}
# {"timestamp":"2025-12-07T09:33:00","direction":"FLAT","reason":"market_choppy","neutral_prob":0.65}
```

**What to look for:**
- Signals appearing every 1-3 minutes (normal)
- Mix of BUY, SELL, FLAT (expected)
- Timestamps advancing monotonically (no time travel)
- buy_prob between 0 and 1 (valid range)

---

### Monitor 2: Feature Logging

```bash
# Watch feature log (for training data)
tail -f feature_log.csv

# Expected output:
# timestamp,user,label,buy_prob,alpha,tradeable,is_flat,tick_count,[features...]
# 2025-12-07T09:30:00,USER,FLAT,0.68,0.0,True,False,45,18505,18490,18480,55,...
# 2025-12-07T09:31:00,USER,FLAT,0.42,0.0,True,True,38,18510,18495,18485,48,...
```

**What to look for:**
- Columns present: timestamp, label, buy_prob, features
- Labels mostly FLAT (at start of day)
- buy_prob ranging 0 to 1
- No missing values

---

### Monitor 3: System Health

```bash
# Check memory usage
watch -n 5 'ps aux | grep "python run_main" | grep -v grep'

# Expected output (should be stable):
# USER   PID  %CPU %MEM    VSZ   RSS COMMAND
# user   123  0.5  1.2  850000 45000 python run_main.py

# Memory should stabilize after 5-10 minutes
# If VSZ keeps growing, there's a leak (check Fix #1 applied)
```

**Alert:** If memory > 500 MB RSS, restart automation

---

### Monitor 4: Error Checking

```bash
# Watch for errors
tail -f logs/main_event_loop.log | grep -i "error\|warning\|exception"

# Expected output:
# (None or very few errors initially)

# If errors appear:
# • "WebSocket connection failed" → API issue
# • "Insufficient funds" → Account issue
# • "Feature mismatch" → Model issue
```

**Action:** If errors, see Troubleshooting section

---

## POST-MARKET PROCEDURES (3:30 PM+)

### Close Automation Gracefully

```bash
# Option 1: Keyboard interrupt (if running in terminal)
Ctrl+C

# Option 2: Kill gracefully
pkill -SIGTERM -f "python run_main.py"

# Option 3: Systemd stop
sudo systemctl stop nifty-automation.service
```

**Expected:** Process stops within 5 seconds, no errors

---

### Archive Logs and Data

```bash
# Create daily archive
DATE=$(date +%Y%m%d)
mkdir -p logs/archive/$DATE

# Copy today's data
cp feature_log.csv logs/archive/$DATE/feature_log_$DATE.csv
cp trained_models/production/signals.jsonl logs/archive/$DATE/signals_$DATE.jsonl

echo "✅ Daily data archived to logs/archive/$DATE/"
```

---

### Optional: Train Models

```bash
# After market close, optionally retrain models
# (See OPERATIONS_AND_TRAINING_GUIDE.md for detailed training workflow)

# Quick training (if 3+ days of data available)
python offline_train_2min.py --start-date $(date -d "7 days ago" +%Y-%m-%d) --end-date $(date +%Y-%m-%d)
python offline_eval_2min_full.py --input feature_log.csv
```

---

## TROUBLESHOOTING

### Problem: Automation Won't Start

**Error:** `ModuleNotFoundError: No module named 'dhan'`

**Solution:**
```bash
pip install dhan  # Install missing package
python run_main.py
```

---

### Problem: WebSocket Connection Failed

**Error:** `ConnectionError: Failed to connect to WebSocket`

**Solution:**
```bash
# 1. Check network connectivity
ping api.dhan.co

# 2. Verify API credentials
echo $DHAN_ACCESS_TOKEN
echo $DHAN_CLIENT_ID

# 3. Check Dhan API status (may be down for maintenance)
# Wait 5 minutes and retry
```

---

### Problem: Memory Growing Unbounded

**Symptom:** Memory usage increasing from 50 MB → 200 MB → 500 MB

**Solution:**
```bash
# This should be fixed by Priority 1 Fix #1
# Verify fix is applied:
grep -n "staged_map = {}" main_event_loop.py

# If not found, the fix wasn't applied
# See CODE_REVIEW_REPORT.md for implementation details
```

---

### Problem: No Signals Being Generated

**Symptom:** signals.jsonl not being updated, feature_log.csv empty

**Solution:**
```bash
# 1. Verify WebSocket is connected
tail -20 logs/main_event_loop.log | grep -i "websocket\|connected"

# 2. Check if market is open
# NSE market hours: 9:15 AM - 3:30 PM IST
date "+%H:%M IST"  # Should show 9:15 - 15:30

# 3. Verify feature data being generated
tail -5 feature_log.csv | wc -l  # Should show multiple rows

# 4. If still no signals, restart:
sudo systemctl restart nifty-automation.service
```

---

## DAILY CHECKLIST (Print and Keep)

```
Date: ___________    Day: ______________

PRE-MARKET (9:00-9:14 AM):
☐ Check env variables set (DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID)
☐ Verify Python 3.10+ installed
☐ Verify required packages installed
☐ Check directory structure (trained_models/production/)
☐ Verify all Python scripts present
☐ Test Dhan API connectivity
☐ Verify Python syntax valid
☐ Confirm file permissions correct
☐ No existing automation process running

STARTUP (9:15 AM):
☐ Execute: python run_main.py
☐ Verify: WebSocket connected within 10 seconds
☐ Verify: Signals being generated
☐ Monitor: Memory usage stays stable

MARKET HOURS (9:15 AM - 3:30 PM):
☐ Check signals every 15 minutes
☐ Monitor memory usage (should be < 200 MB)
☐ Check logs for errors
☐ Verify feature_log.csv being populated

POST-MARKET (3:30 PM+):
☐ Gracefully stop automation (Ctrl+C or systemctl stop)
☐ Archive logs and signals data
☐ (Optional) Train models if 3+ days available
☐ Verify signals.jsonl and feature_log.csv saved
☐ Prepare for next trading day

NOTES:
_________________________________________________________
_________________________________________________________
_________________________________________________________
```

---

**Created:** December 7, 2025  
**Status:** Ready for use  
**Filename:** STARTUP_CHECKLIST.md
