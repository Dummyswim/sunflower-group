"""
dynamic_tuning.py

Lightweight dynamic threshold tuner for live sessions.
Adjusts a small set of thresholds based on label vs intent outcomes,
with bounded updates and JSON persistence for transparency.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TunerConfig:
    enabled: bool
    path: str
    update_every: int = 5
    min_samples: int = 10
    step: float = 0.01
    alpha: float = 0.15
    keys: tuple = ("FLOW_STRONG_MIN", "LANE_SCORE_MIN", "GATE_MARGIN_THR")


@dataclass
class TunerStats:
    total: int = 0
    correct: int = 0
    false_pos: int = 0  # intent BUY/SELL, label FLAT
    false_neg: int = 0  # intent FLAT, label BUY/SELL


class DynamicTuner:
    """Adjust dynamic thresholds based on label/intent drift."""

    def __init__(self, cfg: TunerConfig) -> None:
        self.cfg = cfg
        self.stats = TunerStats()
        self._tick = 0
        self._last_write = 0.0
        self._last_mtime: Optional[float] = None
        self._overrides: Dict[str, float] = {}
        if self.cfg.enabled:
            self._load_overrides()

    def _load_overrides(self) -> None:
        path = self.cfg.path
        if not path or not os.path.isfile(path):
            return
        try:
            mtime = os.path.getmtime(path)
            if self._last_mtime is not None and mtime <= self._last_mtime:
                return
            raw = json.loads(open(path, "r", encoding="utf-8").read())
            overrides = raw.get("overrides", {})
            if isinstance(overrides, dict):
                self._overrides = {k: float(v) for k, v in overrides.items() if v is not None}
            self._last_mtime = mtime
        except Exception:
            return

    def _save_overrides(self) -> None:
        path = self.cfg.path
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "overrides": self._overrides,
                "stats": {
                    "total": self.stats.total,
                    "correct": self.stats.correct,
                    "false_pos": self.stats.false_pos,
                    "false_neg": self.stats.false_neg,
                },
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception:
            return

    def _apply_overrides(self, dyn_thresholds: Any) -> None:
        if not self._overrides:
            return
        for k, v in self._overrides.items():
            try:
                dyn_thresholds.base[k] = float(v)
                dyn_thresholds.curr[k] = float(v)
            except Exception:
                continue

    def update(
        self,
        *,
        label: str,
        intent: str,
        lane: str,
        regime: str,
        dyn_thresholds: Any,
        logger: Any,
    ) -> None:
        if not self.cfg.enabled or dyn_thresholds is None:
            return
        label = str(label or "").upper()
        intent = str(intent or "").upper()
        if label not in ("BUY", "SELL", "FLAT") or intent not in ("BUY", "SELL", "FLAT"):
            return

        self._load_overrides()
        self._apply_overrides(dyn_thresholds)

        self.stats.total += 1
        if label == intent:
            self.stats.correct += 1
        elif label == "FLAT" and intent in ("BUY", "SELL"):
            self.stats.false_pos += 1
        elif intent == "FLAT" and label in ("BUY", "SELL"):
            self.stats.false_neg += 1

        self._tick += 1
        if self._tick % max(1, int(self.cfg.update_every)) != 0:
            return
        if self.stats.total < max(1, int(self.cfg.min_samples)):
            return

        # Adjust direction: tighten on false positives, loosen on false negatives.
        adjust = 0.0
        if self.stats.false_pos > self.stats.false_neg:
            adjust = +abs(self.cfg.step)
        elif self.stats.false_neg > self.stats.false_pos:
            adjust = -abs(self.cfg.step)
        else:
            return

        updated = {}
        for key in self.cfg.keys:
            try:
                cur = float(dyn_thresholds.curr.get(key))
            except Exception:
                continue
            lo, hi = dyn_thresholds.bounds.get(key, (cur * 0.7, cur * 1.5))
            new_val = float(np.clip(cur + adjust, lo, hi))
            dyn_thresholds.base[key] = new_val
            dyn_thresholds.curr[key] = new_val
            updated[key] = new_val
            self._overrides[key] = new_val

        now = time.time()
        if updated and (now - self._last_write) > 10:
            logger.info(
                "[DYN-TUNE] lane=%s regime=%s adjust=%.4f updates=%s stats=%s/%s fp=%s fn=%s",
                lane,
                regime,
                adjust,
                ",".join(f"{k}={v:.4f}" for k, v in updated.items()),
                self.stats.correct,
                self.stats.total,
                self.stats.false_pos,
                self.stats.false_neg,
            )
            self._save_overrides()
            self._last_write = now
