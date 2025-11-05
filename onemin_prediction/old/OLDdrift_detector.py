"""
Concept drift detection using KS-statistics.
"""
from scipy.stats import ks_2samp
import numpy as np

class DriftDetector:
    def __init__(self, train_features):
        self.train_features = train_features

    def detect(self, live_features):
        drift_stats = {}
        # CHANGED: Handle scalar live values and missing keys safely [[11]]
        for feat in self.train_features:
            try:
                base = np.asarray(self.train_features[feat], dtype=float)
                # If live feature is scalar, wrap into a tiny array for KS
                live_val = live_features.get(feat, None)
                if live_val is None:
                    continue  # skip missing
                live_arr = np.asarray(live_val if hasattr(live_val, '__len__') else [live_val], dtype=float)
                if base.size == 0 or live_arr.size == 0:
                    continue
                stat, pval = ks_2samp(base, live_arr, alternative='two-sided', mode='auto')
                drift_stats[feat] = {'ks_stat': float(stat), 'p_value': float(pval)}
            except Exception:
                # Skip features that cannot be compared
                continue
        return drift_stats
