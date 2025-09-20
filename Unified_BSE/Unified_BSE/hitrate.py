from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, time
import json
import os
import math

@dataclass
class Candidate:
    next_bar_time: datetime
    direction: str                 # 'BUY' | 'SELL' | 'NEUTRAL'
    actionable: bool               # passed all gates?
    rejection_reason: Optional[str]
    mtf_score: float
    breadth: int                   # active_indicators
    weighted_score: float
    macd_hist_slope: float = 0.0
    rsi_value: float = 50.0
    rsi_cross_up: bool = False
    rsi_cross_down: bool = False
    pattern_used: bool = False
    sr_room: str = "UNKNOWN"       # 'AVAILABLE' | 'LIMITED' | 'UNKNOWN'
    regime: str = "UNKNOWN"
    confidence: float = 0.0
    saved_at: datetime = None

def _bucket_mtf(x: float) -> str:
    if x < 0.20: return "MTF<0.20"
    if x < 0.50: return "0.20–0.50"
    if x < 0.65: return "0.50–0.65"
    if x < 0.80: return "0.65–0.80"
    return ">=0.80"

def _bucket_breadth(b: int) -> str:
    if b <= 2: return "0–2"
    if b == 3: return "3"
    return ">=4"

def _bucket_strength(s: float) -> str:
    a = abs(s)
    if a < 0.10: return "|s|<0.10"
    if a < 0.30: return "0.10–0.30"
    if a < 0.60: return "0.30–0.60"
    return ">=0.60"

def _bucket_slope(v: float) -> str:
    if v <= -1e-6: return "slope_down"
    if v >=  1e-6: return "slope_up"
    return "slope_flat"

def _bucket_rsi(value: float, up: bool, down: bool) -> str:
    if up: return "rsi_50_cross_up"
    if down: return "rsi_50_cross_down"
    return "rsi_above50" if value >= 50 else "rsi_below50"

def _bucket_tod(ts: datetime) -> str:
    t = ts.time()
    if t <= time(10, 15): return "open"
    if t <  time(14, 30): return "mid"
    return "close"

class HitRateTracker:
    def __init__(self, jsonl_path: str = "logs/hitrate.jsonl"):
        self.stats: Dict[Tuple[str, ...], list] = defaultdict(lambda: [0, 0, 0, 0])
        self.pending: Dict[datetime, Candidate] = {}
        self.jsonl_path = jsonl_path
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    def save_candidate(self, c: Candidate):
        self.pending[c.next_bar_time] = c


    def resolve_bar(self, next_bar_time: datetime, open_price: float, close_price: float, logger=None):
        
        c = self.pending.pop(next_bar_time, None)
        if not c:
            return

        dir_up = (c.direction or "NEUTRAL").upper()
        base_dir = dir_up.replace("STRONG_", "")
        if base_dir not in ("BUY", "SELL"):
            # persist for coverage analytics too
            self._dump_jsonl(c, correct=None, open_price=open_price, close_price=close_price)
            return

        # sanitize open/close
        try:
            open_price = float(open_price)
        except Exception:
            open_price = 0.0
        try:
            close_price = float(close_price)
        except Exception:
            close_price = 0.0
        if not math.isfinite(open_price):
            open_price = 0.0
        if not math.isfinite(close_price):
            close_price = 0.0


        up = (close_price - open_price) > 0
        correct = (base_dir == "BUY" and up) or (base_dir == "SELL" and not up)


        key = (
            _bucket_mtf(c.mtf_score),
            _bucket_breadth(c.breadth),
            _bucket_strength(c.weighted_score),
            _bucket_slope(c.macd_hist_slope),
            _bucket_rsi(c.rsi_value, c.rsi_cross_up, c.rsi_cross_down),
            "pattern_used" if c.pattern_used else "pattern_off",
            c.sr_room or "UNKNOWN",
            _bucket_tod(next_bar_time)
        )
        row = self.stats[key]
        row[0] += 1
        row[1] += int(correct)
        if c.actionable:
            row[2] += 1
            row[3] += int(correct)

        self._dump_jsonl(c, correct=correct, open_price=open_price, close_price=close_price)
        if logger:
            tag = "[ACT]" if c.actionable else "[CAL]"
            logger.info(f"{tag} {'✓' if correct else '✗'} Next-bar: {c.direction} | "
                        f"move={close_price-open_price:+.2f} | mtf={c.mtf_score:.2f} | "
                        f"active={c.breadth}/6 | strength={c.weighted_score:+.3f} | "
                        f"slope={c.macd_hist_slope:+.6f} | rsi={c.rsi_value:.1f} "
                        f"{'(upX)' if c.rsi_cross_up else '(dnX)' if c.rsi_cross_down else ''} | "
                        f"pattern={'Y' if c.pattern_used else 'N'} | sr={c.sr_room} | "
                        f"rej={c.rejection_reason or '-'} | conf={c.confidence:.0%}")

    def _dump_jsonl(self, c: Candidate, correct: Optional[bool], open_price: float, close_price: float):
        rec = {
            "next_bar_time": c.next_bar_time.isoformat(),
            "direction": c.direction,
            "actionable": c.actionable,
            "rejection_reason": c.rejection_reason,
            "mtf_score": c.mtf_score,
            "breadth": c.breadth,
            "weighted_score": c.weighted_score,
            "macd_hist_slope": c.macd_hist_slope,
            "rsi_value": c.rsi_value,
            "rsi_cross_up": c.rsi_cross_up,
            "rsi_cross_down": c.rsi_cross_down,
            "pattern_used": c.pattern_used,
            "sr_room": c.sr_room,
            "regime": c.regime,
            "confidence": c.confidence,
            "open": open_price,
            "close": close_price,
            "correct": correct,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def report(self, logger, min_samples: int = 5):
        logger.info("=" * 60)
        logger.info("===== HIT-RATE BUCKETS =====")
        # 1) MTF × Breadth
        agg = defaultdict(lambda: [0, 0])
        for (mtf, br, *_), (tot, cor, _, _) in self.stats.items():
            k = (mtf, br); agg[k][0] += tot; agg[k][1] += cor
        for (mtf, br), (tot, cor) in sorted(agg.items()):
            if tot >= min_samples:
                acc = 100.0 * cor / max(1, tot)
                logger.info(f"[MTF={mtf} | Breadth={br}] total={tot} acc={acc:.1f}%")


        # 2) Strength bands
        band = defaultdict(lambda: [0, 0])
        for key, (tot, cor, _, _) in self.stats.items():
            strength = key[2]  # strength bucket at index 2
            band[strength][0] += tot
            band[strength][1] += cor
        for strength, (tot, cor) in sorted(band.items()):
            if tot >= min_samples:
                acc = 100.0 * cor / max(1, tot)
                logger.info(f"[Strength={strength}] total={tot} acc={acc:.1f}%")


        # 3) Momentum slope
        slope = defaultdict(lambda: [0, 0])
        for key, (tot, cor, _, _) in self.stats.items():
            # key layout: (mtf, breadth, strength, slope, rsi, pattern, sr_room, tod)
            sl = key[3]  # FIX: extract the slope bucket (index 3)
            slope[sl][0] += tot
            slope[sl][1] += cor  # FIX: use index [1] for correct count
        for sl, (tot, cor) in sorted(slope.items()):
            if tot >= min_samples:
                acc = 100.0 * cor / max(1, tot)
                logger.info(f"[Slope={sl}] total={tot} acc={acc:.1f}%")

        logger.info("===== END OF HIT-RATE BUCKETS =====")
        logger.info("=" * 60)