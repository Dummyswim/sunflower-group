# Model Performance Analysis: Train/Eval Window Impact

**Date:** December 7, 2025  
**Summary:** Changed eval window from training period to out-of-sample period, revealing significant overfitting.

---

## Executive Summary

Your model shows **EXCELLENT performance on training data** (80.8% accuracy) but **POOR performance on new data** (57.2% accuracy). This is a textbook case of **overfitting + distribution shift**.

### Key Finding
- **Same model, same training code**
- **Different eval windows = Drastically different results**
- This tells us the model learned training data patterns, not general market patterns

---

## Performance Comparison

### Setup 1: Eval on Training Period (In-Sample)
```
TRAIN: Jan 1, 2024 - Oct 29, 2025
EVAL:  Jan 1, 2024 - Oct 29, 2025  ← Same data!
```

**Results: PERFECT calibration**
```
Overall accuracy:           80.8% ✅
High-confidence (≥0.70):    99.1% ✅

Accuracy by confidence:
├─ 0.50-0.60: 64.8% (n=7,880)
├─ 0.60-0.70: 91.6% (n=4,803)
├─ 0.70-0.80: 98.8% (n=2,339)  ← Almost perfect!
├─ 0.80-0.90: 99.9% (n=757)     ← Essentially perfect!
└─ 0.90-1.00: 100.0% (n=120)    ← All correct!
```

### Setup 2: Eval on Test Period (Out-of-Sample)
```
TRAIN: Jan 1, 2024 - Oct 29, 2025
EVAL:  Nov 1, 2025 - Dec 5, 2025  ← New data!
```

**Results: Model fails completely**
```
Overall accuracy:           57.2% ✗
High-confidence (≥0.70):    51.8% ✗

Accuracy by confidence:
├─ 0.50-0.60: 56.4% (n=110)
├─ 0.60-0.70: 61.1% (n=95)
├─ 0.70-0.80: 59.5% (n=37)      ← Worse than random!
├─ 0.80-0.90: 36.8% (n=19)      ← FAILS at high confidence!
└─ High-conf subset: 51.8%       ← No better than coin flip
```

---

## Root Cause Analysis

### 1. OVERFITTING (Primary Issue)

Your model learned **memorized patterns** in training data, not generalizable trading signals.

**Evidence:**
- Perfect calibration in-sample (100% on 0.90-1.00 bin)
- Complete calibration failure out-of-sample (37% on 0.80-0.90 bin)
- When model says "very confident" on NEW data, it's WRONG 63% of the time

**Why this happens:**
- 103 features + 168,916 training samples = High risk of overfitting
- XGBoost with default parameters can overfit easily on tabular data
- No regularization control mentioned

### 2. DISTRIBUTION SHIFT (Secondary Issue)

Market behavior changed from Oct 29 → Nov 1.

**Evidence:**
- Different probability calibration curves
- Directional accuracy: 80.8% → 57.2% (23.6% drop)
- High-confidence predictions become WORSE (paradoxical)

**Likely causes:**
- Volatility regime change
- Different time-of-year patterns (Halloween effect, holiday season)
- Different market participants
- Economic news between Oct-Nov
- FII/DII flows or seasonal patterns in Indian markets

---

## Detailed Failure Mode Analysis

### Why High-Confidence Predictions FAIL

**In-sample (Oct 29):**
- Model says "90%+ confident" → Correct 100% of the time
- Model has learned exact decision boundaries

**Out-of-sample (Nov 1+):**
- Model says "90%+ confident" → Correct only 37% of the time
- Decision boundaries don't apply to new market regime
- Model is OVERCONFIDENT on out-of-distribution data

This is dangerous! The model generates false confidence signals.

---

## Q-Model Gating Analysis

From your latest run, the Q-model gating results:

```
Baseline accuracy (all directional):           0.572
After gating (margin≥0.150, q_hat≥0.550):     0.476  ← WORSE!
```

**Finding:** The gating makes things worse, which means:
- Q-model is also overfitted on training data
- It's selecting the WRONG subset of trades
- Cannot be trusted for out-of-sample filtering

---

## What This Means

### ✅ Good News
1. **Model architecture is sound** - Perfect performance in-sample proves the approach works
2. **Features have signal** - For Oct 2024-2025 period, features predictive
3. **Training procedure is correct** - Achieved 80.8% is not luck, it's learning something

### ❌ Bad News
1. **Cannot deploy this model** - Out-of-sample accuracy too low
2. **Cannot trust confidence scores** - High confidence = False signal
3. **Need distribution shift handling** - Market changed Nov 1

---

## Solution Strategy

### Phase 1: Verify Overfitting (Quick - 30 min)

**Test 1: Walk-forward validation**
```
├─ Train on Jan 2024 - Aug 2025
├─ Test on Sep 2025
├─ Record accuracy
├─ Train on Jan 2024 - Sep 2025
├─ Test on Oct 2025
├─ Record accuracy
└─ Compare to in-sample accuracy
```

Expected: Accuracy should decline gradually (80% → 70% → 60%)

**Test 2: Check regularization**
```bash
# Look at XGBoost parameters
grep -A 10 "xgb_params = {" offline_train_2min.py
```

Expected: Should see `max_depth`, `learning_rate`, `reg_alpha`, `reg_lambda`

### Phase 2: Fix Overfitting (Medium - 2-4 hours)

**Option A: Add Regularization**
```python
xgb_params = {
    'max_depth': 5,              # Reduce from default 6
    'learning_rate': 0.01,       # Reduce from default 0.1
    'subsample': 0.8,            # Use 80% samples per tree
    'colsample_bytree': 0.8,     # Use 80% features per tree
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
    'min_child_weight': 5,       # Require more samples in leaves
}
```

**Option B: Feature selection**
- Use only top 40-50 of 103 features
- Reduces model complexity
- Often improves generalization

**Option C: Early stopping**
```python
# Stop training when validation accuracy stops improving
eval_set = [(X_val, y_val)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, 
              early_stopping_rounds=10)
```

### Phase 3: Handle Distribution Shift (Medium - 2-4 hours)

**Option 1: Retraining schedule**
- Retrain model every 1-2 months
- Keep last 6 months of data as training set
- Automatically adapts to market changes

**Option 2: Online learning**
- Update model weights on live trades
- Continuous adaptation to new regime
- More complex but more robust

**Option 3: Ensemble approach**
- Train separate models for different seasons/regimes
- Use market indicators to switch between models
- Hedge against distribution shift

---

## Immediate Action Items

### Today (30 minutes)
- [ ] Run walk-forward validation test (3 splits)
- [ ] Check XGBoost parameter regularization
- [ ] Save results to `MODEL_VALIDATION_RESULTS.md`

### This Week (4-8 hours)
- [ ] Implement Option A or B (regularization or feature selection)
- [ ] Retrain with regularization
- [ ] Test on hold-out set (e.g., Nov 1-15)
- [ ] Compare before/after accuracy

### This Month (Ongoing)
- [ ] Design retraining schedule
- [ ] Implement automated retraining
- [ ] Monitor accuracy on daily basis

---

## Key Takeaways

| Metric | In-Sample | Out-of-Sample | Implication |
|--------|-----------|---------------|-------------|
| Overall Accuracy | 80.8% | 57.2% | 23.6% degradation |
| High-Conf Accuracy | 99.1% | 51.8% | Overconfidence! |
| Confidence Calibration | Perfect | Broken | Distribution shift |
| Model Stability | ✅ | ❌ | Cannot deploy yet |

**Bottom Line:** Your model is overfitting training data. Before deploying, you need:
1. **Regularization** to reduce complexity
2. **Validation strategy** to detect overfitting early
3. **Retraining schedule** to handle distribution shift

---

## Code Changes Needed

### Change 1: Add Regularization to XGBoost
In `offline_train_2min.py`, find where `xgb_params` is defined and update:

```python
xgb_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cpu',
    
    # Regularization parameters (NEW)
    'max_depth': 5,              # Reduce tree depth
    'learning_rate': 0.01,       # Lower learning rate
    'subsample': 0.8,            # Use 80% samples
    'colsample_bytree': 0.8,     # Use 80% features
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
    'min_child_weight': 5,       # Require more samples in leaves
}
```

### Change 2: Implement Walk-Forward Validation
Create new file `validate_overfitting.py`:

```python
import pandas as pd
import numpy as np
from offline_train_2min import train_models_2min, make_trade_outcome_label
from intraday_cache_manager import get_cache_manager

# Split 1: Train on Jan-Aug 2025, test on Sep 2025
# Split 2: Train on Jan-Sep 2025, test on Oct 2025
# Split 3: Train on Jan-Oct 2025, test on Nov 2025

splits = [
    ('2024-01-01', '2025-08-31', '2025-09-01', '2025-09-30'),
    ('2024-01-01', '2025-09-30', '2025-10-01', '2025-10-31'),
    ('2024-01-01', '2025-10-29', '2025-11-01', '2025-12-05'),
]

for train_start, train_end, test_start, test_end in splits:
    print(f"\nSplit: Train {train_start}→{train_end}, Test {test_start}→{test_end}")
    # Run training and evaluation
    # Record accuracy
```

---

## Expected Outcomes

If you implement regularization:
- **In-sample accuracy:** 80.8% → 75-78% (small decline, better generalization)
- **Out-of-sample accuracy:** 57.2% → 65-70% (significant improvement!)
- **Calibration:** Closer match between in-sample and out-of-sample

This would be healthy overfitting reduction.

---

## Questions to Answer

1. **What changed on Nov 1?** (Political event? FII flows? Volatility?)
2. **Is this temporary or permanent?** (Test Nov 1-15 vs Nov 16-30)
3. **Do other symbols show same pattern?** (Is it market-wide?)
4. **When was model last retrained?** (Older model = more drift)

---

## Next Steps

1. **Confirm overfitting** via walk-forward validation
2. **Add regularization** to reduce complexity
3. **Set retraining schedule** (monthly?)
4. **Monitor live performance** daily
5. **Document distribution shifts** for future reference

Once you fix overfitting, your model should be production-ready! 🚀
