
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
    direction: str
    actionable: bool
    rejection_reason: Optional[str]
    mtf_score: float
    breadth: int
    weighted_score: float
    macd_hist_slope: float = 0.0
    rsi_value: float = 50.0
    rsi_cross_up: bool = False
    rsi_cross_down: bool = False
    pattern_used: bool = False
    sr_room: str = "UNKNOWN"
    regime: str = "UNKNOWN"
    confidence: float = 0.0
    oi_signal: Optional[str] = None
    oi_change_pct: float = 0.0
    saved_at: Optional[datetime] = None
    promoted: bool = False
    # NEW: multi-horizon + liberal shadow classification
    horizon: str = "5m"
    liberal_direction: Optional[str] = None  # 'BUY'|'SELL'|'NEUTRAL' or None


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
    def __init__(self, jsonl_path: str = None, config=None):
        self.stats: Dict[Tuple[str, ...], list] = defaultdict(lambda: [0, 0, 0, 0])

        self.pending: Dict[Tuple[str, datetime], Candidate] = {}
        # Rotation config
        self._cfg_rotate = bool(getattr(config, 'hitrate_rotate_daily', True)) if config else True
        self._cfg_base = str(getattr(config, 'hitrate_base_path', "logs/hitrate")) if config else "logs/hitrate"
        self._cfg_keep_days = int(getattr(config, 'hitrate_keep_days', 60)) if config else 60
        self._cfg_symlink = bool(getattr(config, 'hitrate_symlink_latest', True)) if config else True
        
        # Liberal shadow stats (summary logs)
        self._lib_trials = 0
        self._lib_scored = 0
        self._lib_wins = 0
        
                
        # Path resolve
        if jsonl_path:
            self._current_path = jsonl_path
            os.makedirs(os.path.dirname(self._current_path), exist_ok=True)
        else:
            os.makedirs(os.path.dirname(self._cfg_base), exist_ok=True)
            from datetime import date
            self._current_path = self._path_for_day(date.today())
            self._update_symlink_latest()

    def _path_for_day(self, d) -> str:
        return f"{self._cfg_base}_{d.strftime('%Y%m%d')}.jsonl"

    def _roll_if_new_day(self):
        if not self._cfg_rotate:
            return
        from datetime import date
        today = date.today()
        if not os.path.basename(self._current_path).endswith(today.strftime("%Y%m%d") + ".jsonl"):
            self._current_path = self._path_for_day(today)
            self._update_symlink_latest()
            self._purge_old_files()

    def _update_symlink_latest(self):
        if not self._cfg_symlink:
            return
        try:
            link_path = f"{self._cfg_base}.jsonl"
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(os.path.basename(self._current_path), link_path)
        except Exception:
            pass

    def _purge_old_files(self):
        try:
            import glob
            from datetime import datetime as _dt, timedelta
            cutoff = _dt.now() - timedelta(days=self._cfg_keep_days)
            for fp in glob.glob(f"{self._cfg_base}_*.jsonl"):
                try:
                    mtime = _dt.fromtimestamp(os.path.getmtime(fp))
                    if mtime < cutoff:
                        os.remove(fp)
                except Exception:
                    continue
        except Exception:
            pass

    def _append_jsonl(self, rec: dict):
        self._roll_if_new_day()
        try:
            with open(self._current_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            with open(self._current_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")






    def save_candidate(self, c: Candidate):
        # Do not overwrite an existing candidate for the same horizon+bar
        key = (str(getattr(c, 'horizon', '5m')), c.next_bar_time)
        if key in self.pending:
            return
        c.promoted = bool(c.actionable)
        self.pending[key] = c



    def resolve_bar(self, next_bar_time: datetime, open_price: float, close_price: float, logger=None, horizon: str = "5m"):
        c = self.pending.pop((str(horizon), next_bar_time), None)
        if not c:
            # Backward fallback for older saves (pre-horizon)
            c = self.pending.pop(('5m', next_bar_time), None)
        if not c:
            return
        

        dir_up = (c.direction or "NEUTRAL").upper()
        base_dir = dir_up.replace("STRONG_", "")
        # Liberal evaluation target (BUY/SELL?) saved at candidate time
        lib_dir = (getattr(c, 'liberal_direction', None) or "NEUTRAL").upper()

        # sanitize open/close (as-is)
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

        move = close_price - open_price
        # Same micro-move skip rule
        


        try:
            pts_min = 1.0
            pct_min = 0.0003
            if str(horizon).lower() == "1m":
                pts_min = 0.5
                pct_min = 0.0001
            tiny = abs(move) < max(pts_min, pct_min * max(1.0, open_price))
        except Exception:
            tiny = False


        lib_correct = None  # default
        if tiny:
            # liberal trial counted as trial-only
            if lib_dir in ("BUY","SELL"):
                self._lib_trials += 1
                
            self._dump_jsonl(c, correct=None, open_price=open_price, close_price=close_price, liberal_correct=None, scored=False, skipped_reason="micro_move")
          
            if logger:
                logger.info(f"[HR] Skipped micro-move scoring (Δ={move:+.2f}) for {base_dir} ({horizon})")
                
            return


        # Strict correctness (unchanged)
        if base_dir not in ("BUY", "SELL"):
            # persist NEUTRAL for coverage
            
            self._dump_jsonl(c, correct=None, open_price=open_price, close_price=close_price, liberal_correct=None, scored=False, skipped_reason="neutral")
            return

        up = (move > 0)
        correct = (base_dir == "BUY" and up) or (base_dir == "SELL" and not up)


        # Liberal correctness (if BUY/SELL)
        if lib_dir in ("BUY","SELL"):
            self._lib_trials += 1
            self._lib_scored += 1
            lib_correct = (lib_dir == "BUY" and up) or (lib_dir == "SELL" and not up)
            if lib_correct:
                self._lib_wins += 1

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
        # row indices: 0=total, 1=correct, 2=act_total, 3=act_correct
        row[0] += 1
        row[1] += int(bool(correct))
        if c.actionable:
            row[2] += 1
            row[3] += int(bool(correct))


        
        self._dump_jsonl(c, correct=correct, open_price=open_price, close_price=close_price, liberal_correct=lib_correct, scored=True, skipped_reason=None)
        
        if logger:
            tag = "[ACT]" if c.actionable else "[CAL]"
            logger.info(f"{tag} {'✓' if correct else '✗'} Next-bar({horizon}): {c.direction} | move={close_price-open_price:+.2f} | mtf={c.mtf_score:.2f} | active={c.breadth}/6 | strength={c.weighted_score:+.3f} | slope={c.macd_hist_slope:+.6f} | rsi={c.rsi_value:.1f} { '(upX)' if c.rsi_cross_up else '(dnX)' if c.rsi_cross_down else '' } | pattern={'Y' if c.pattern_used else 'N'} | sr={c.sr_room} | rej={c.rejection_reason or '-'} | conf={c.confidence:.0%}")
            


    # add liberal fields, horizon to JSONL
    def _dump_jsonl(self, c: Candidate, correct: Optional[bool], open_price: float, close_price: float, liberal_correct: Optional[bool] = None, scored: Optional[bool] = None, skipped_reason: Optional[str] = None):

        rec = {
            "next_bar_time": c.next_bar_time.isoformat(),
            "horizon": str(getattr(c, 'horizon', '5m')),
            "direction": c.direction,
            "liberal_direction": getattr(c, 'liberal_direction', None),
            "actionable": c.actionable,
            "promoted": bool(getattr(c, 'promoted', c.actionable)),
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
            "oi_signal": c.oi_signal,
            "oi_change_pct": c.oi_change_pct,
            "open": open_price,
            "close": close_price,
            "correct": correct,
            "liberal_correct": liberal_correct,
            "scored": bool(scored) if scored is not None else (correct is not None),
            "skipped_reason": skipped_reason
        }
        self._append_jsonl(rec)





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
        



        # Liberal shadow summary
        if self._lib_trials > 0:
            wr = 100.0 * self._lib_wins / max(1, self._lib_scored)
            logger.info(f"[LIBERAL] trials={self._lib_trials} scored={self._lib_scored} win_rate={wr:.1f}%")
            
        logger.info("=" * 60)        