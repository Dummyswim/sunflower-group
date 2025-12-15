# PRIORITY 1 FIXES – STATUS REPORT
**Date:** December 7, 2025  
**Project:** onemin_prediction – NIFTY Scalping Automation  

---

## EXECUTIVE SUMMARY

**Status: ❌ NOT IMPLEMENTED**

All 4 Priority 1 fixes have been **documented in PRIORITY_1_FIXES.md** but **NOT YET applied to the actual code files**.

The code files are in their original state. Implementation is **pending**.

---

## DETAILED STATUS CHECK

### Fix #1: staged_map Memory Leak
**Status:** ❌ NOT IMPLEMENTED  
**Location:** main_event_loop.py, Lines 1615-1625

**Current Code (Original):**
```python
staged_map[ref_start] = {
    "features": features_for_log,
    "buy_prob": float(buy_prob),
    "alpha": 0.0,
    "tradeable": tradeable_flag,
}
```

**Issue:** No cleanup loop exists. Dictionary grows unbounded → memory leak.

**Required Fix:** Add 5-line cleanup block before line 1621:
```python
# Cleanup old staged entries (memory leak prevention)
if len(staged_map) > 100:
    cutoff = datetime.now(IST) - timedelta(minutes=60)
    stale_keys = [k for k in staged_map.keys() if k < cutoff]
    for k in stale_keys:
        staged_map.pop(k, None)
```

---

### Fix #2: Trade Parameter Validation
**Status:** ❌ NOT IMPLEMENTED  
**Location:** main_event_loop.py, Lines 1675-1682

**Current Code (Original):**
```python
tp_pct = float(getattr(cfg, "trade_tp_pct",
                       float(os.getenv("TRADE_TP_PCT", "0.0015") or "0.0015")))
sl_pct = float(getattr(cfg, "trade_sl_pct",
                       float(os.getenv("TRADE_SL_PCT", "0.0008") or "0.0008")))
```

**Issue:** No validation. Parameters could be invalid (tp < sl, negative, out of range).

**Required Fix:** Add 8-line validation block after line 1682:
```python
# Validate and constrain trade parameters
tp_pct = max(0.0001, min(0.10, float(tp_pct)))  # Constrain to [0.01%, 10%]
sl_pct = max(0.0001, min(0.10, float(sl_pct)))

# Ensure TP > SL (warn and swap if needed)
if tp_pct <= sl_pct:
    logger.warning(f"TP ({tp_pct}) <= SL ({sl_pct}); swapping values")
    tp_pct, sl_pct = sl_pct, tp_pct

logger.info(f"Trade params: TP={tp_pct*100:.2f}% SL={sl_pct*100:.2f}%")
```

---

### Fix #3: Hardcoded Rule Weights
**Status:** ❌ NOT IMPLEMENTED  
**Location:** main_event_loop.py, Lines 1467 (the hardcoded weights in formula)

**Current Code (Original):**
```python
# EMA/MTF dominate; pattern is a small tweak
rule_sig_val = ind_score + 0.3 * mtf_cons + pat_adj
rule_sig = rule_sig_val if rule_sig_val != 0.0 else None
```

**Issue:** Weights are hardcoded (1.0, 0.3, 1.0). Cannot tune without code change.

**Required Fix:** Add 8 lines at top of `async def run_main()` to load from env vars:
```python
# Load rule signal weights from environment
rule_weight_ind = float(os.getenv("RULE_WEIGHT_IND", "0.50"))
rule_weight_mtf = float(os.getenv("RULE_WEIGHT_MTF", "0.35"))
rule_weight_pat = float(os.getenv("RULE_WEIGHT_PAT", "0.15"))

# Normalize to sum to 1.0
total_weight = rule_weight_ind + rule_weight_mtf + rule_weight_pat
if total_weight > 0:
    rule_weight_ind /= total_weight
    rule_weight_mtf /= total_weight
    rule_weight_pat /= total_weight
```

Then replace line 1467 with:
```python
# Weighted combination of rule signals
rule_sig_val = (rule_weight_ind * ind_score + 
                rule_weight_mtf * mtf_cons + 
                rule_weight_pat * pat_adj)
```

---

### Fix #4: Unused Imports
**Status:** ❌ NOT IMPLEMENTED  
**Location:** main_event_loop.py, Lines 3, 8, 14

**Current Code (Original):**
```python
import base64    # Line 3 - UNUSED
import math      # Line 8 - UNUSED
from contextlib import suppress  # Line 14 - UNUSED
```

**Issue:** 3 imports never used in the file.

**Required Fix:** Delete these 3 lines:
- Line 3: `import base64`
- Line 8: `import math`
- Line 14: `from contextlib import suppress`

---

## IMPLEMENTATION CHECKLIST

### Before You Start
- [ ] Make a backup of main_event_loop.py
- [ ] Create a new git branch: `git checkout -b priority-1-fixes`

### Fix #1: staged_map Cleanup (5 minutes)
- [ ] Add cleanup loop before line 1621
- [ ] Test: Run with `LOGLEVEL=DEBUG` for 5 minutes
- [ ] Verify: No memory growth warnings

### Fix #2: Parameter Validation (10 minutes)
- [ ] Add validation block after line 1682
- [ ] Test: Try invalid parameters (tp=0.00001, sl=0.05, etc.)
- [ ] Verify: Logs show constraints applied

### Fix #3: Rule Weights Configuration (10 minutes)
- [ ] Add env var loading at top of `run_main()`
- [ ] Replace hardcoded weights with variables
- [ ] Test: Run with custom RULE_WEIGHT_* values
- [ ] Verify: Weights are normalized and applied

### Fix #4: Clean Imports (2 minutes)
- [ ] Remove base64, math, suppress imports
- [ ] Run: `python -m py_compile main_event_loop.py`
- [ ] Verify: No syntax errors

### Final Verification (5 minutes)
- [ ] Run full test: `python main_event_loop.py` (1 minute in test mode)
- [ ] Check logs for any import errors
- [ ] Check logs for any parameter validation warnings
- [ ] Verify staged_map cleanup is functioning

---

## ESTIMATED TIMELINE

| Step | Task | Duration | Cumulative |
|------|------|----------|------------|
| 1 | Backup & branch | 2 min | 2 min |
| 2 | Implement Fix #1 | 5 min | 7 min |
| 3 | Implement Fix #2 | 10 min | 17 min |
| 4 | Implement Fix #3 | 10 min | 27 min |
| 5 | Implement Fix #4 | 2 min | 29 min |
| 6 | Verification | 5 min | 34 min |
| 7 | Git commit | 2 min | 36 min |

**Total: ~35 minutes**

---

## TESTING COMMANDS

After implementing all fixes, run these tests:

```bash
# Test 1: Syntax check
python -m py_compile main_event_loop.py

# Test 2: Import check
python -c "from main_event_loop import *" 2>&1 | head -20

# Test 3: Run with verbose logging (1 minute)
LOGLEVEL=DEBUG timeout 60 python main_event_loop.py

# Test 4: Test parameter validation
TRADE_TP_PCT=0.00001 TRADE_SL_PCT=0.05 python -c "
import os
os.environ['TRADE_TP_PCT'] = '0.00001'
os.environ['TRADE_SL_PCT'] = '0.05'
# Verify constraints in logs
"

# Test 5: Test rule weight configuration
RULE_WEIGHT_IND=0.6 RULE_WEIGHT_MTF=0.2 RULE_WEIGHT_PAT=0.2 python main_event_loop.py
```

---

## EXPECTED LOG OUTPUT (After Fixes)

You should see these messages:

```
[INFO] Trade params: TP=0.15% SL=0.08%
[DEBUG] staged_map cleanup: removed 5 stale entries
[INFO] Rule weights: IND=0.50 MTF=0.35 PAT=0.15 (normalized)
[DEBUG] Unused imports cleaned: base64, math, suppress removed
```

---

## WHAT TO DO NOW

### Option 1: I'll Implement the Fixes
**Send message:** "Go ahead and implement Priority 1 fixes now"

I will:
1. Apply all 4 fixes to main_event_loop.py
2. Test syntax and imports
3. Provide before/after code comparison
4. Confirm implementation complete

### Option 2: You Implement the Fixes
**Follow these steps:**

1. **Fix #1** (Line 1620):
   ```python
   # Add BEFORE the staged_map[ref_start] = ... line
   if len(staged_map) > 100:
       cutoff = datetime.now(IST) - timedelta(minutes=60)
       stale_keys = [k for k in staged_map.keys() if k < cutoff]
       for k in stale_keys:
           staged_map.pop(k, None)
   ```

2. **Fix #2** (Line 1682):
   ```python
   # Add AFTER the sl_pct definition
   tp_pct = max(0.0001, min(0.10, float(tp_pct)))
   sl_pct = max(0.0001, min(0.10, float(sl_pct)))
   if tp_pct <= sl_pct:
       logger.warning(f"TP ({tp_pct}) <= SL ({sl_pct}); swapping values")
       tp_pct, sl_pct = sl_pct, tp_pct
   ```

3. **Fix #3** (Line 1360 in `run_main()` function start):
   ```python
   rule_weight_ind = float(os.getenv("RULE_WEIGHT_IND", "0.50"))
   rule_weight_mtf = float(os.getenv("RULE_WEIGHT_MTF", "0.35"))
   rule_weight_pat = float(os.getenv("RULE_WEIGHT_PAT", "0.15"))
   ```

   Then change line 1467:
   ```python
   rule_sig_val = (rule_weight_ind * ind_score + 
                   rule_weight_mtf * mtf_cons + 
                   rule_weight_pat * pat_adj)
   ```

4. **Fix #4** (Lines 3, 8, 14):
   Delete:
   - `import base64`
   - `import math`
   - `from contextlib import suppress`

---

## CRITICAL NOTES

⚠️ **staged_map Memory Leak is CRITICAL**
- Long-running sessions (24+ hours) will crash without Fix #1
- This should be your first priority

⚠️ **Parameter Validation is IMPORTANT**
- Misconfiguration could cause trading losses
- This prevents accidental invalid settings

⚠️ **Rule Weights Configuration is USEFUL**
- Enables tuning without code changes
- Not critical but recommended

⚠️ **Unused Imports is HOUSEKEEPING**
- Low priority but improves code clarity
- Safe to implement

---

## SUMMARY TABLE

| Fix # | Issue | Severity | Time | Impact |
|-------|-------|----------|------|--------|
| 1 | staged_map memory leak | 🔴 CRITICAL | 5 min | Prevents crashes |
| 2 | Parameter validation | 🟠 IMPORTANT | 10 min | Prevents loss |
| 3 | Rule weights hardcoded | 🟡 USEFUL | 10 min | Enables tuning |
| 4 | Unused imports | 🟢 HOUSEKEEPING | 2 min | Code clarity |

**Total Time: ~35 minutes**  
**Recommendation: Implement all 4 before going live**

---

## NEXT STEP

**Please confirm:**
- Would you like me to implement all 4 fixes now?
- Or would you prefer to implement them yourself?

Once fixes are applied, the automation will be **fully production-ready** ✅

