"""
Minimal online logistic regression for 1m prediction.
NaN/Inf-safe, numerically stable, with L2 and learning-rate decay.
"""
from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OnlineLogit:
    """
    Minimal online logistic regression for 1m prediction.
    - Features: micro-only (imb, imbS, slp_avg, slpS, momΔ, stdΔ, drift, nL, nS, z_sig)
    - Numeric guards: NaN→0, Inf→0; gradient clip; L2 reg; LR decay
    - Predict: returns p(up) in [0,1]
    - Update: one SGD step with L2
    """
    
    def __init__(self, n_features: int, lr: float = 0.03, l2: float = 0.0005, clip: float = 3.0):
        self.n = int(max(1, n_features))
        self.w = np.zeros(self.n, dtype=float)
        self.lr0 = float(lr)
        self.l2 = float(l2)
        self.clip = float(clip)
        self.t = 0  # update counter

    @staticmethod
    def _sigm(x: float) -> float:
        """Stable sigmoid"""
        if x >= 0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        z = np.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _sanitize(x: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with 0"""
        x = np.asarray(x, dtype=float)
        x[~np.isfinite(x)] = 0.0
        return x

    def predict_proba(self, x: np.ndarray) -> float:
        """Predict probability of class 1"""
        x = self._sanitize(x)
        if x.shape[0] != self.n:
            x = np.resize(x, self.n)
        s = float(np.dot(self.w, x))
        p = self._sigm(s)
        return max(1e-6, min(1.0 - 1e-6, p))

    def update(self, x: np.ndarray, y: int) -> None:
        """
        One online SGD step. y in {0,1}
        """
        try:
            x = self._sanitize(x)
            if x.shape[0] != self.n:
                x = np.resize(x, self.n)
            p = self.predict_proba(x)
            
            # Gradient of log-loss w.r.t w: (p - y) * x
            g = (p - float(y)) * x
            
            # Gradient clip
            g = np.clip(g, -self.clip, self.clip)
            
            # Decayed learning rate
            self.t += 1
            lr = self.lr0 / np.sqrt(1.0 + 0.05 * self.t)
            
            # L2 regularization
            self.w -= lr * (g + self.l2 * self.w)
            
            if self.t % 50 == 0:
                logger.info("[NM-ML] step=%d | lr=%.4f | ||w||=%.4f | last_p=%.3f | y=%d",
                            self.t, lr, float(np.linalg.norm(self.w)), p, int(y))
        except Exception as e:
            logger.debug("[NM-ML] update skipped: %s", e)
