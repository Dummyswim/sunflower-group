import json
import glob
import os
import math
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

def _jeffreys_winrate(wins: int, total: int) -> Tuple[float, float, float]:
    """ Jeffreys prior Beta(0.5, 0.5): posterior mean p and ~68% CI via normal approx.
    Returns (p, ci_low, ci_high) in [0,1]. """
    if total <= 0:
        return 0.5, 0.25, 0.75
    alpha = wins + 0.5
    beta = (total - wins) + 0.5
    p = alpha / (alpha + beta)
    var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    sd = math.sqrt(var) if var > 0 else 0.0
    lo = max(0.0, p - sd)
    hi = min(1.0, p + sd)
    return float(p), float(lo), float(hi)

def _parse_ts(iso_or_str: str) -> Optional[datetime]:
    """ Robust ISO parser; returns IST. If parsing fails, returns None. """
    try:
        if not iso_or_str:
            return None
        s = str(iso_or_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None: # Treat naive as IST
            return dt.replace(tzinfo=IST)
        return dt.astimezone(IST)
    except Exception:
        return None

def _bucket(x: float, edges: Tuple[float, ...], labels: Optional[Tuple[str, ...]] = None) -> str:
    """ Generic numeric bucketing helper. """
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if labels: # If explicit labels provided, map by edges
        for i, e in enumerate(edges):
            if v < e:
                return labels[i]
        return labels[-1]
    # Default string labels
    for e in edges:
        if v < e:
            return f"<{e}"
    return f">={edges[-1]}" if edges else "NA"

def _bucket_mtf(x: float) -> str:
    return _bucket(x, (0.20, 0.50, 0.65, 0.80), ("MTF<0.20", "0.20–0.50", "0.50–0.65", "0.65–0.80", ">=0.80"))

def _bucket_breadth(b: int) -> str:
    try:
        bi = int(b)
    except Exception:
        bi = 0
    if bi <= 2: return "0–2"
    if bi == 3: return "3"
    return ">=4"

def _bucket_strength(s: float) -> str:
    try:
        a = abs(float(s))
    except Exception:
        a = 0.0
    if a < 0.10: return "|s|<0.10"
    if a < 0.30: return "0.10–0.30"
    if a < 0.60: return "0.30–0.60"
    return ">=0.60"

def _bucket_slope(v: float) -> str:
    try:
        x = float(v)
    except Exception:
        x = 0.0
    if x <= -1e-6: return "slope_down"
    if x >= 1e-6: return "slope_up"
    return "slope_flat"

def _bucket_rsi(value: float, up: bool, down: bool) -> str:
    try:
        v = float(value)
    except Exception:
        v = 50.0
    up = bool(up)
    down = bool(down)
    if up: return "rsi_50_cross_up"
    if down: return "rsi_50_cross_down"
    return "rsi_above50" if v >= 50.0 else "rsi_below50"

def _bucket_tod(ts: Optional[datetime]) -> str:
    if not ts:
        return "mid"
    hhmm = ts.hour * 60 + ts.minute
    if hhmm <= (10 * 60 + 15): # <= 10:15
        return "open"
    if hhmm < (14 * 60 + 30): # < 14:30
        return "mid"
    return "close"

def make_setup_key(rec: Dict) -> Tuple[str, ...]:
    """ Build a compact, stable setup signature from a hitrate JSONL record (or analyzer rec).
    Compatible with your current hitrate.py fields and analyzer rec composition.

    Expected fields (with fallbacks):
      - direction (BUY/SELL/NEUTRAL or with STRONG_ prefix)
      - mtf_score, breadth, weighted_score
      - macd_hist_slope, rsi_value, rsi_cross_up, rsi_cross_down
      - sr_room
      - oi_signal (optional; default 'neutral')
      - next_bar_time (ISO) or 'tod' provided by caller

    Returns a tuple of strings that index the conditional tables.
    """
    try:
        side = str(rec.get("direction", "NEUTRAL")).upper().replace("STRONG_", "")
        mtf = float(rec.get("mtf_score", 0.0) or 0.0)
        breadth = int(rec.get("breadth", 0) or 0)
        strength = float(rec.get("weighted_score", 0.0) or 0.0)
        slope = float(rec.get("macd_hist_slope", 0.0) or 0.0)
        rsi = float(rec.get("rsi_value", 50.0) or 50.0)
        rsi_up = bool(rec.get("rsi_cross_up", False))
        rsi_dn = bool(rec.get("rsi_cross_down", False))
        sr_room = str(rec.get("sr_room", "UNKNOWN") or "UNKNOWN")
        oi_sig = str(rec.get("oi_signal", "neutral") or "neutral")

        # Time-of-day bucket: prefer explicit 'tod', else derive from next_bar_time
        tod = str(rec.get("tod") or "").lower()
        if tod not in ("open", "mid", "close"):
            ts = _parse_ts(rec.get("next_bar_time", "")) if "next_bar_time" in rec else None
            tod = _bucket_tod(ts)

        k_mtf = _bucket_mtf(mtf)
        k_brd = _bucket_breadth(breadth)
        k_str = _bucket_strength(strength)
        k_slope = _bucket_slope(slope)
        k_rsi = _bucket_rsi(rsi, rsi_up, rsi_dn)

        return (side, k_mtf, k_brd, k_str, k_slope, k_rsi, sr_room, oi_sig, tod)
    except Exception as e:
        logger.debug(f"[EVIDENCE] make_setup_key error: {e}")
        # Fallback minimal key
        return (str(rec.get("direction", "NEUTRAL")).upper().replace("STRONG_", ""), "NA", "NA", "NA", "NA", "NA", str(rec.get("sr_room","UNKNOWN")), str(rec.get("oi_signal","neutral")), "mid")

class SetupStats:
    """ Tiny in-memory conditional tables:
    - recent (default: last 56 days)
    - long (default: last 180 days)
    Reads daily-rotated hitrate JSONL: {jsonl_base}_YYYYMMDD.jsonl
    Also reads a plain {jsonl_base}.jsonl if present (symlink or today’s file).
    """

    def __init__(self, jsonl_base: str = "logs/hitrate", recent_days: int = 56, long_days: int = 180):
        self.jsonl_base = jsonl_base
        self.recent_days = int(recent_days)
        self.long_days = int(long_days)
        self.recent: Dict[Tuple[str, ...], Tuple[int, int]] = {}
        self.long: Dict[Tuple[str, ...], Tuple[int, int]] = {}
        # Track mtimes to support cheap update_from_tail()
        self._file_mtimes: Dict[str, float] = {}

    def _iter_files(self) -> List[str]:
        """
        Return a sorted list of hitrate files to read.
        Pattern: {base}_*.jsonl plus optional {base}.jsonl (symlink or plain).
        """
        pat = f"{self.jsonl_base}_*.jsonl"
        files = sorted(glob.glob(pat))
        plain = f"{self.jsonl_base}.jsonl"
        # Include plain file if exists and not already included by pattern
        if os.path.exists(plain) and plain not in files:
            files.append(plain)
        return files

    def build(self) -> None:
        """
        Build recent and long tables from current files.
        """
        now = datetime.now(IST)
        self.recent.clear()
        self.long.clear()

        files = self._iter_files()
        if not files:
            logger.info("[EVIDENCE] No hitrate files found under base=%s", self.jsonl_base)
            
            return

        read = 0
        recent_added = 0
        long_added = 0

        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue

                        # Only evaluate when correctness is present or explicitly None (we accumulate totals either way)
                        t = _parse_ts(rec.get("next_bar_time", "")) or now
                        age_days = (now - t).days if t else 9999

                        # Determine correctness (skip None from wins; still counts in totals)
                        corr = rec.get("correct", None)
                        is_win = bool(corr) if corr is not None else False

                        key = make_setup_key(rec)

                        # Long table
                        tot, wins = self.long.get(key, (0, 0))
                        self.long[key] = (tot + 1, wins + (1 if is_win else 0))
                        long_added += 1

                        # Recent table
                        if age_days <= self.recent_days:
                            r_tot, r_wins = self.recent.get(key, (0, 0))
                            self.recent[key] = (r_tot + 1, r_wins + (1 if is_win else 0))
                            recent_added += 1

                        read += 1
            except Exception as e:
                logger.debug(f"[EVIDENCE] File read skipped ({fp}): {e}")

        # Update mtime snapshot
        self._file_mtimes = {fp: (os.path.getmtime(fp) if os.path.exists(fp) else 0.0) for fp in files}

        logger.info("[EVIDENCE] Build complete: files=%d, rows=%d | recent_keys=%d items, long_keys=%d items",
                    len(files), read, len(self.recent), len(self.long))
        




    def weekly_summary(self, logger, days: int = 7) -> None:
        """
        Compare 1m vs 5m over last `days` by sr_room and regime (based on JSONL recs).
        Visible high-signal logs only (no tables).
        """
        try:
            now = datetime.now(IST)
            files = self._iter_files()
            cutoff = now - timedelta(days=int(days))
            agg = {"1m": {"tot":0,"win":0}, "5m": {"tot":0,"win":0}}
            by_sr = {"1m": {}, "5m": {}}
            by_reg = {"1m": {}, "5m": {}}

            for fp in files:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        for line in f:
                            rec = json.loads(line.strip())
                            ts = _parse_ts(rec.get("next_bar_time", "")) or now
                            if ts < cutoff:
                                continue
                            hz = str(rec.get("horizon", "5m"))
                            cor = rec.get("correct", None)
                            if hz not in agg:
                                continue
                            agg[hz]["tot"] += 1
                            if cor is True: agg[hz]["win"] += 1
                            sr = str(rec.get("sr_room", "UNKNOWN"))
                            rg = str(rec.get("regime", "UNKNOWN"))
                            by_sr[hz].setdefault(sr, [0,0])
                            by_reg[hz].setdefault(rg, [0,0])
                            by_sr[hz][sr][0] += 1
                            by_reg[hz][rg][0] += 1
                            
                            if cor is True:
                                by_sr[hz][sr][1] += 1
                                by_reg[hz][rg][1] += 1


                except Exception:
                    continue

            def _wr(t,w): 
                return 100.0 * w / max(1,t)

            logger.info("[WEEKLY] Horizon totals (last %d days):", int(days))
            for hz in ("5m","1m"):
                t,w = agg[hz]["tot"], agg[hz]["win"]
                logger.info("  %s: total=%d win_rate=%.1f%%", hz, t, _wr(t,w))
            logger.info("[WEEKLY] By SR room:")
            for hz, mp in by_sr.items():
                for k,(t,w) in mp.items():
                    if t >= 10:
                        logger.info("  %s | SR=%s: total=%d win_rate=%.1f%%", hz, k, t, _wr(t,w))
            logger.info("[WEEKLY] By regime:")
            for hz, mp in by_reg.items():
                for k,(t,w) in mp.items():
                    if t >= 10:
                        logger.info("  %s | regime=%s: total=%d win_rate=%.1f%%", hz, k, t, _wr(t,w))
            
        except Exception as e:
            logger.debug(f"[WEEKLY] summary error: {e}")






    def _files_changed(self) -> bool:
        """
        Detect if any hitrate file has changed mtime since last build.
        """
        files = self._iter_files()
        for fp in files:
            try:
                m = os.path.getmtime(fp)
            except Exception:
                m = 0.0
            if fp not in self._file_mtimes:
                return True
            if abs(m - self._file_mtimes.get(fp, 0.0)) > 1e-6:
                return True
        # Also detect deletions
        for fp in list(self._file_mtimes.keys()):
            if fp not in files:
                return True
        return False

    def update_from_tail(self) -> None:
        """
        Cheap refresh: rebuild only if any file mtime changed.
        """
        try:
            if self._files_changed():
                logger.info("[EVIDENCE] Change detected in hitrate files → rebuilding tables")
                
                self.build()
        except Exception as e:
            logger.debug(f"[EVIDENCE] Tail update skipped: {e}")

    def get(self, key: Tuple[str, ...], window: str = "recent") -> Tuple[float, float, float, int]:
        """
        Return (p, lo, hi, n) for the given key from 'recent' or 'long' table.
        p, lo, hi are in [0,1]. n is total observations for that key in the requested window.
        """
        table = self.recent if str(window).lower() == "recent" else self.long
        tot, wins = table.get(key, (0, 0))
        p, lo, hi = _jeffreys_winrate(wins, tot)
        return p, lo, hi, tot
