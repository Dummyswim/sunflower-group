# Quick Fix: Regularization Implementation Guide

**Goal:** Reduce overfitting from 80.8% (in-sample) to closer match with 57.2% (out-of-sample)

---

## Step 1: Examine Current XGBoost Parameters

First, check what parameters are currently set:

```bash
grep -n "xgb_params\|max_depth\|learning_rate\|subsample\|colsample" offline_train_2min.py | head -30
```

Look for something like:
```python
xgb_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    ...
}
```

---

## Step 2: Add Regularization Parameters

**If regularization is missing**, add these parameters to `xgb_params`:

```python
xgb_params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cpu',
    
    # REGULARIZATION (Add these):
    'max_depth': 5,              # Limit tree depth (default is 6)
    'learning_rate': 0.01,       # Slower learning (default is 0.1)
    'subsample': 0.8,            # Use 80% of samples per tree
    'colsample_bytree': 0.8,     # Use 80% of features per tree
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
    'min_child_weight': 5,       # Require more samples in leaves
}
```

---

## Step 3: Test Different Regularization Strengths

Create `test_regularization.py`:

```python
#!/usr/bin/env python3
"""Test different regularization strengths."""

import os
import sys
from offline_train_2min import main as train_main
from offline_eval_2min import main as eval_main

# Test configurations
configs = [
    {
        'name': 'Baseline (No regularization)',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
    },
    {
        'name': 'Light regularization',
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.01,
        'reg_lambda': 0.1,
    },
    {
        'name': 'Medium regularization',
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    },
    {
        'name': 'Heavy regularization',
        'max_depth': 4,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 10.0,
    },
]

results = []

for config in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"{'='*60}")
    
    # Set environment variables
    os.environ['TRAIN_START_DATE'] = '2024-01-01 09:30:00'
    os.environ['TRAIN_END_DATE'] = '2025-10-29 15:15:00'
    
    # Train model with this config
    # Note: You'll need to modify offline_train_2min.py to accept these as params
    # For now, this is a template
    
    print(f"  max_depth: {config['max_depth']}")
    print(f"  learning_rate: {config['learning_rate']}")
    print(f"  subsample: {config['subsample']}")
    print(f"  colsample_bytree: {config['colsample_bytree']}")
    print(f"  reg_alpha: {config['reg_alpha']}")
    print(f"  reg_lambda: {config['reg_lambda']}")
    
    # Evaluate on test set
    os.environ['EVAL_START_DATE'] = '2025-11-01'
    os.environ['EVAL_END_DATE'] = '2025-12-05'
    
    # Extract accuracy from logs
    # results.append({
    #     'config': config['name'],
    #     'in_sample_acc': ...,
    #     'out_sample_acc': ...
    # })

print("\n\nResults Summary:")
print("="*60)
# for r in results:
#     print(f"{r['config']}: In={r['in_sample_acc']:.1%}, Out={r['out_sample_acc']:.1%}")
```

---

## Step 4: Expected Results After Regularization

| Parameter | Current | With Regularization | Effect |
|-----------|---------|-------------------|--------|
| In-Sample Accuracy | 80.8% | ~75-78% | Small decline (acceptable) |
| Out-Sample Accuracy | 57.2% | ~65-70% | Large improvement! |
| Overfitting Gap | 23.6% | ~5-10% | Much better generalization |

---

## Step 5: Implement Early Stopping (Optional)

If XGBoost supports eval_set, add:

```python
# In offline_train_2min.py, when training XGBoost:

eval_set = [(X_val, y_val)]  # Validation set

xgb_model = xgb.XGBClassifier(
    **xgb_params,
    eval_metric='logloss',
)

xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    verbose=True
)
```

---

## Step 6: Retraining Schedule

Add to your training documentation:

```bash
# Retrain every month with last 12 months of data
MONTHLY_RETRAINING=true

# Or retrain every week if you notice accuracy drop
WEEKLY_RETRAINING=false

# Monitor out-of-sample accuracy daily
DAILY_MONITORING=true
```

---

## Quick Comparison: Before vs After

### Before Regularization
```
Model trained on: Jan 2024 - Oct 2025
Evaluated on:     Oct 2024 (training period)
├─ Accuracy: 80.8% ✅
├─ Confidence calibration: Perfect ✅
└─ Issue: Model overfitted!

Evaluated on:     Nov 2025 (test period)
├─ Accuracy: 57.2% ❌
├─ Confidence calibration: Broken ❌
└─ Issue: Cannot generalize!
```

### After Regularization (Expected)
```
Model trained on: Jan 2024 - Oct 2025
Evaluated on:     Oct 2024 (training period)
├─ Accuracy: 75-78% ✅
├─ Confidence calibration: Good ✅
└─ Issue: Slightly lower but more honest

Evaluated on:     Nov 2025 (test period)
├─ Accuracy: 65-70% ✅
├─ Confidence calibration: Decent ✅
└─ Issue: Much better generalization!
```

---

## Troubleshooting

**Q: After regularization, in-sample accuracy drops too much (60%)**
- You over-regularized
- Try lighter regularization (subsample=0.9, reg_lambda=0.1)

**Q: Out-of-sample accuracy still doesn't improve (still 57%)**
- Distribution shift may be real market change
- Try retraining on more recent data only
- Or accept that this market period is hard to predict

**Q: High-confidence predictions still fail (37% on 0.80-0.90 bin)**
- Regularization won't fix this alone
- Need to:
  1. Add market regime detection
  2. Retrain monthly
  3. Use ensemble of models

---

## Implementation Checklist

- [ ] Examine current `xgb_params` in `offline_train_2min.py`
- [ ] Add regularization parameters
- [ ] Test on validation set
- [ ] Compare in-sample vs out-of-sample accuracy
- [ ] If acceptable, update production model
- [ ] Set up monthly retraining schedule
- [ ] Monitor accuracy daily
- [ ] Document any market regime changes

---

## Key Insight

Your model learned training data, not market patterns. Regularization reduces model complexity so it:
1. Performs slightly worse on training data (acceptable)
2. Performs much better on new data (goal!)
3. Has better probability calibration (more trustworthy)

This is the right trade-off for a trading model!
