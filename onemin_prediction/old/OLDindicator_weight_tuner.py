"""
Adaptive indicator weight tuner using 1m hitrate logs (volume-free).
Reads JSONL hitrate logs and updates weights for:
- micro_slope
- imbalance (micro_imbalance)
- mean_drift (micro_drift)

Note: EMA trend weight is kept anchored (no direct metric in logs).
"""
import asyncio
import json
from collections import deque
from typing import Dict, Optional

def _sign(x: float) -> int:
    try:
        if x > 0: return 1
        if x < 0: return -1
        return 0
    except Exception:
        return 0

class IndicatorWeightTuner:
    def __init__(
        self,
        base_weights: Dict[str, float],
        lr: float = 0.05,
        ema_anchor: float = 0.35,
        min_w: float = 0.05,
        max_w: float = 0.80,
        max_lines: int = 8000
    ):
        self.weights = dict(base_weights)  # copy
        self.lr = float(lr)
        self.ema_anchor = float(ema_anchor)
        self.min_w = float(min_w)
        self.max_w = float(max_w)
        self.max_lines = int(max_lines)
        # These three are tunable from logs
        self.tunable_keys = ['micro_slope', 'imbalance', 'mean_drift']

    async def update_from_file(self, path: str) -> Optional[Dict[str, float]]:
        """
        Read last N lines from hitrate JSONL and update weights.
        Returns new weights if update succeeded, else None.
        """
        try:
            # Offload I/O to thread to avoid blocking event loop
            lines = await asyncio.to_thread(self._tail_lines, path, self.max_lines)
            if not lines:
                return None

            # Accumulators for accuracy
            counts = {'micro_slope': 0, 'imbalance': 0, 'mean_drift': 0}
            corrects = {'micro_slope': 0, 'imbalance': 0, 'mean_drift': 0}

            for ln in lines:
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                # Filter to 1m horizon entries with valid OHLC
                if obj.get('horizon') != '1m':
                    continue
                open_px = obj.get('open', None)
                close_px = obj.get('close', None)
                if open_px is None or close_px is None:
                    continue
                gt = _sign(float(close_px) - float(open_px))  # ground truth direction for the minute

                # Pull micro features if present
                m_slope = obj.get('micro_slope', None)
                m_imb = obj.get('micro_imbalance', None)
                m_drift = obj.get('micro_drift', None)

                # micro_slope
                if m_slope is not None:
                    sig = _sign(float(m_slope))
                    if sig != 0 and gt != 0:
                        counts['micro_slope'] += 1
                        if sig == gt:
                            corrects['micro_slope'] += 1
                # imbalance
                if m_imb is not None:
                    sig = _sign(float(m_imb))
                    if sig != 0 and gt != 0:
                        counts['imbalance'] += 1
                        if sig == gt:
                            corrects['imbalance'] += 1
                # mean_drift (from micro_drift)
                if m_drift is not None:
                    sig = _sign(float(m_drift))
                    if sig != 0 and gt != 0:
                        counts['mean_drift'] += 1
                        if sig == gt:
                            corrects['mean_drift'] += 1

            # Compute accuracies with Laplace smoothing
            acc = {}
            for k in self.tunable_keys:
                n = counts[k]
                c = corrects[k]
                acc[k] = (c + 1.0) / (n + 2.0) if n >= 0 else 0.5

            # Normalize accuracies into a simplex share for tunables
            s = sum(acc.values()) or 1.0
            acc_norm = {k: (acc[k] / s) for k in acc}

            # Blend into current weights (EMA-style), keep EMA trend anchored
            new_w = dict(self.weights)
            for k_map, k_w in [('micro_slope', 'micro_slope'),
                               ('imbalance', 'imbalance'),
                               ('mean_drift', 'mean_drift')]:
                prev = float(self.weights.get(k_w, 0.0))
                target = float(acc_norm.get(k_map, prev))
                blended = (1.0 - self.lr) * prev + self.lr * target
                new_w[k_w] = float(max(self.min_w, min(self.max_w, blended)))

            # Anchor ema_trend around ema_anchor, then renormalize all 4 to sum to 1
            new_w['ema_trend'] = float(max(self.min_w, min(self.max_w, self.ema_anchor)))
            tot = sum(new_w.values()) or 1.0
            for k in new_w:
                new_w[k] = float(new_w[k] / tot)

            self.weights = new_w
            return dict(self.weights)
        except Exception:
            return None

    def _tail_lines(self, path: str, max_lines: int):
        try:
            dq = deque(maxlen=max_lines)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    dq.append(line.strip())
            return list(dq)
        except Exception:
            return []
