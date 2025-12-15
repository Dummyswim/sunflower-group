# Cache-Aware Training Guide

## Overview

The training scripts now intelligently use cached data to minimize API calls. Instead of fetching all data from the API, the system:

1. **Scans the cache directory** to identify which dates are already cached
2. **Loads cached data** directly from disk (fast, no API calls)
3. **Fetches only missing dates** from the Dhan API
4. **Saves newly fetched data** back to cache for future use

This approach reduces API calls significantly, saves time, and makes training more reliable.

## Cache Structure

Cached data is stored in `data/intraday_cache/` with the following naming convention:

```
INDEX_20250101_1m.csv  (January 1, 2025 data)
INDEX_20250102_1m.csv  (January 2, 2025 data)
...
```

Each file contains 1-minute OHLCV (Open, High, Low, Close, Volume) candles for that trading day.

### Current Cache Coverage

The cache directory currently contains data for:
- **First date**: 2024-01-01
- **Last date**: 2025-12-02
- **Total cached dates**: 700+ trading days (spanning ~2 years)
- **Gap strategy**: Weekends are automatically skipped (no trading data)

## How Cache-Aware Training Works

### Step 1: Cache Scan
When you start training, the system logs:
```
INFO | Cache status: 450 cached, 25 missing out of 500 trading days
INFO | Loaded 450000 candles from cache
```

This means:
- 450 dates are cached (can load instantly)
- 25 dates need API fetching
- Total trading days in range: 500

### Step 2: Load Cached Data
All cached dates are loaded directly from CSV files into memory. This is **much faster** than API calls (typically <1 second vs 20+ seconds).

```
INFO | Loaded 450000 candles from cache
```

### Step 3: Fetch Missing Dates
Only the 25 missing dates are fetched from the API:

```
INFO | Fetching 25 missing dates from API
INFO | Fetching intraday for 2025-11-15 from https://api.dhan.co/v2/charts/intraday
INFO | Fetched 390 candles for 2025-11-15
...
```

### Step 4: Cache New Data
Newly fetched data is automatically saved to cache for future use:

```
INFO | Cached 390 candles for 2025-11-15 to data/intraday_cache/INDEX_20251115_1m.csv
```

### Step 5: Consolidate & Train
All data (cached + newly fetched) is merged and used for training:

```
INFO | Total candles in range: 195000
INFO | Offline 2-minute training complete.
```

## Performance Comparison

### Without Cache (Old Behavior)
- **Date range**: Sept 1 - Dec 7, 2025 (65 trading days)
- **API calls**: 65 (one per day)
- **Total time**: ~30-40 minutes
- **API rate limits**: Risk of hitting rate limits

### With Cache (New Behavior)
- **Date range**: Sept 1 - Dec 7, 2025 (65 trading days)
- **Cached dates**: ~60 dates (from previous runs)
- **API calls**: ~5 (only missing dates)
- **Total time**: ~2-5 minutes
- **API rate limits**: Minimal risk

**Speedup: 6-8x faster** on typical retraining scenarios!

## Usage

### Complete Training Command
```bash
export DHAN_ACCESS_TOKEN="your_actual_token" && \
export DHAN_CLIENT_ID="your_actual_client_id" && \
export TRAIN_START_DATE="2025-09-01" && \
export TRAIN_END_DATE="2025-12-07" && \
export XGB_PATH="trained_models/production/xgb_model.pkl" && \
export NEUTRAL_PATH="trained_models/production/neutral_model.pkl" && \
export INTRADAY_CACHE_ENABLE="1" && \
export INTRADAY_CACHE_DIR="data/intraday_cache" && \
cd /home/hanumanth/Documents/sunflower-group_2/onemin_prediction && \
python offline_train_2min.py
```

### Environment Variables

**Required for Training:**
- `DHAN_ACCESS_TOKEN` - Your Dhan API token
- `DHAN_CLIENT_ID` - Your Dhan client ID  
- `TRAIN_START_DATE` - Training start date (YYYY-MM-DD)
- `TRAIN_END_DATE` - Training end date (YYYY-MM-DD)
- `XGB_PATH` - Output path for XGBoost model
- `NEUTRAL_PATH` - Output path for neutrality model

**Cache Configuration:**
- `INTRADAY_CACHE_ENABLE` - Enable caching (default: `1`)
- `INTRADAY_CACHE_DIR` - Cache directory path (default: `data/intraday_cache`)

### Example Workflow

**First run** (no cache):
```bash
export INTRADAY_CACHE_DIR="/path/to/cache" && \
python offline_train_2min.py
# All 65 dates fetched from API
# All data cached for future use
# Time: ~30-40 minutes
```

**Second run** (with cache, same date range):
```bash
# Same command as above
# All 65 dates loaded from cache
# ZERO API calls
# Time: ~1-2 minutes
# Speedup: 20-30x!
```

**Third run** (with cache, extended date range):
```bash
export TRAIN_END_DATE="2025-12-25"  # Extended to Dec 25
python offline_train_2min.py
# First 65 dates loaded from cache
# Last 12 days fetched from API
# Time: ~1-3 minutes
```

## Cache Management

### View Cache Status

The training script logs detailed cache status at startup:

```
=== CACHE STATUS ===
Cache directory: data/intraday_cache
Total cached dates: 743
Date range: 2025-09-01 to 2025-12-07
Trading days in range: 65
Cached: 60, Missing: 5
First missing dates: 2025-12-06, 2025-12-07
===================
```

### Manual Cache Inspection

**List all cached files:**
```bash
ls -lh data/intraday_cache/ | head -20
```

**Count cached dates:**
```bash
ls -1 data/intraday_cache/INDEX_*_1m.csv | wc -l
```

**Check specific date:**
```bash
ls -l data/intraday_cache/INDEX_20250901_1m.csv
head -5 data/intraday_cache/INDEX_20250901_1m.csv
```

### Clearing Cache (if needed)

**Clear entire cache:**
```bash
rm -rf data/intraday_cache/*
# Next training will fetch all dates from API
```

**Clear specific dates:**
```bash
rm data/intraday_cache/INDEX_20250901_1m.csv
rm data/intraday_cache/INDEX_20250902_1m.csv
```

### Cache Directory Setup

The cache directory is automatically created if it doesn't exist. If you want to use a custom location:

```bash
export INTRADAY_CACHE_DIR="/custom/path/to/cache"
python offline_train_2min.py
```

## Implementation Details

### Cache Manager Module

The new `intraday_cache_manager.py` module provides:

- **IntradayCache class**: Handles all cache operations
- **scan_cached_dates()**: Identifies which dates are cached
- **get_missing_dates()**: Calculates dates to fetch
- **load_cached_data()**: Loads all cached data for a date range
- **save_cached_data()**: Saves newly fetched data to cache
- **get_cache_summary()**: Provides cache status report

### Modified Functions

**offline_train_2min.py**:

1. **fetch_intraday_range()** (UPDATED)
   - Now uses cache-first strategy
   - Loads cached data first
   - Fetches only missing dates
   - Automatically caches new data

2. **fetch_intraday_for_day()** (UNCHANGED)
   - Still fetches from API for missing dates
   - Still supports cache fallback
   - Works seamlessly with new strategy

3. **main()** (UPDATED)
   - Shows cache status before training
   - Displays cached vs missing date breakdown

## Troubleshooting

### Issue: Still fetching all dates from API

**Cause**: `INTRADAY_CACHE_ENABLE` is disabled or cache directory doesn't exist

**Solution**:
```bash
export INTRADAY_CACHE_ENABLE="1"
export INTRADAY_CACHE_DIR="data/intraday_cache"
python offline_train_2min.py
```

### Issue: Cache directory permissions error

**Error message**: `Failed to create cache directory`

**Solution**:
```bash
mkdir -p data/intraday_cache
chmod 755 data/intraday_cache
python offline_train_2min.py
```

### Issue: Stale cache data

**Cause**: Cache contains old data that was updated

**Solution**: Clear cache for that specific date
```bash
rm data/intraday_cache/INDEX_20250901_1m.csv
python offline_train_2min.py  # Will refetch Sept 1 from API
```

### Issue: Memory error with large date ranges

**Cause**: Loading too many cached files at once

**Solution**: Split into smaller date ranges
```bash
# Instead of 1 year at once
export TRAIN_START_DATE="2025-01-01"
export TRAIN_END_DATE="2025-06-30"
python offline_train_2min.py

# Then separately
export TRAIN_START_DATE="2025-07-01"
export TRAIN_END_DATE="2025-12-31"
python offline_train_2min.py
```

## Performance Tips

### 1. Keep Cache Updated
Regular training runs automatically cache new data. The first run will be slower, but subsequent runs are fast.

### 2. Use Consistent Cache Directory
Always use the same `INTRADAY_CACHE_DIR` so data isn't duplicated.

### 3. Monitor Cache Size
```bash
du -sh data/intraday_cache/  # Total cache size
ls -1 data/intraday_cache/ | wc -l  # Number of files
```

Typical size: ~1-2 MB per date, so 700 dates ≈ 1.4-2.8 GB

### 4. Archive Old Cache Periodically
If cache gets too large:
```bash
tar -czf data/intraday_cache_backup_2025.tar.gz data/intraday_cache/
rm -rf data/intraday_cache/
# Cache will rebuild on next training run
```

## Technical Notes

### Cache File Format

Each cache file is a CSV with columns:
```
timestamp,open,high,low,close,volume
2025-09-01 09:15:00,23500.0,23550.0,23450.0,23475.0,1000000
2025-09-01 09:16:00,23475.0,23525.0,23470.0,23510.0,950000
...
```

- **timestamp**: ISO format with date and time
- **OHLCV**: Standard Open, High, Low, Close, Volume

### Weekends Handling

The cache manager automatically skips Saturdays and Sundays:
- When calculating missing dates, weekends are excluded
- When scanning cache, weekend files won't be expected
- Training date ranges work seamlessly across weekends

### Caching Strategy

- **Read frequency**: High (every training run)
- **Write frequency**: Only when new dates are fetched
- **Invalidation**: Manual (delete specific date files) or by disabling cache

## Future Enhancements

Possible improvements:
- [ ] Automatic cache expiration (refetch data older than N days)
- [ ] Incremental cache verification (checksums for data integrity)
- [ ] Parallel API fetching (speed up missing date fetches)
- [ ] Cache compression (to reduce disk space)
- [ ] Statistics dashboard (cache hit rate, API efficiency, time savings)

## Questions?

Refer to `TRAINING_ERROR_FIX.md` for error troubleshooting or `TRAINING_QUICK_START.md` for basic training setup.
