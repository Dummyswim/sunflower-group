#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dhan v2 Live Candlestick Pattern Detector & Next-Candle Predictor

Enhancements included:
  (a) Rolling backtest priors (100-bar hit-rate per pattern, self-adapting)
  (b) ATR-based confidence scaling
  (c) Chart export with pattern highlight (auto-sent to Telegram if configured)
  (d) CSV replay mode for offline testing (no live feed required)
  (e) Rolling accuracy logging per pattern whenever a pattern fires

Requirements (install):
  pip install pandas numpy matplotlib TA-Lib websocket-client requests

Notes:
- Connects to Dhan v2 WebSocket feed and decodes the QUOTE packet (ResponseCode=4).
- Aggregates 1-minute OHLC using LTT (exchange epoch seconds) to avoid clock skew.
- Detects ALL TA-Lib pattern functions dynamically.
- Maintains a 100-bar rolling hit-rate per pattern and uses that as the prior.
- Predicts next candle direction via weighted vote + small momentum + ATR scaling.
- When patterns fire, exports a PNG of the last ~30 bars and (optionally) sends to Telegram.
- CSV replay lets you test detections/predictions/prior-learning offline.

CLI examples:

  # Live mode with Telegram charts
  python dhani_fixed.py \
      --client-id YOUR_DHAN_CLIENT_ID \
      --access-token YOUR_DHAN_ACCESS_TOKEN \
      --instruments IDX_I:13 NSE_EQ:11536 \
      --telegram-token YOUR_TELEGRAM_BOT_TOKEN \
      --telegram-chat-id 123456789

  # Offline test replay (CSV columns: time,open,high,low,close)
  python dhani_fixed.py --replay-csv sample.csv --log DEBUG
"""

import argparse
import collections
import json
import logging
import os
import ssl
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
from matplotlib import ticker

try:
    import talib
except Exception as e:
    talib = None

try:
    import websocket  # websocket-client
except Exception:
    websocket = None

try:
    import requests
except Exception:
    requests = None


# --------------------------
# Dhan v2 constants
# --------------------------

EXCHANGE_SEGMENT_MAP = {
    "IDX_I": 0, "NSE_EQ": 1, "NSE_FNO": 2, "NSE_CURRENCY": 3,
    "BSE_EQ": 4, "BSE_FNO": 5, "BSE_CURRENCY": 6, "MCX": 7,
}

REQ_SUB_QUOTE = 17
RES_QUOTE = 4
RES_FEED_DISCONNECT = 50

# Header: <B H B I> = code(1), msg_len(2), exchange(1), security_id(4)
HDR_FMT = "<B H B I"
HDR_SIZE = struct.calcsize(HDR_FMT)

# Quote payload (42 bytes):
# ltp f32, ltq i16, ltt i32, atp f32, vol i32, tot_sell i32, tot_buy i32,
# day_open f32, day_close f32, day_high f32, day_low f32
QUOTE_FMT = "<f h i f i i i f f f f"
QUOTE_SIZE = struct.calcsize(QUOTE_FMT)


# --------------------------
# Utilities
# --------------------------

def ts_to_minute(epoch: int) -> datetime:
    """Convert exchange epoch seconds to a minute bucket (UTC)."""
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.replace(second=0, microsecond=0)


def ensure_talib():
    if talib is None:
        raise RuntimeError("TA-Lib is required. Install with: pip install TA-Lib")


# --------------------------
# Pattern Engine with rolling priors
# --------------------------

class PatternEngine:
    """
    Maintains a rolling (window) hit-rate per TA-Lib pattern function; this hit-rate
    is used as the pattern's probability prior. If no history yet, falls back to default.
    """
    def __init__(self, default_prob: float = 0.55, window: int = 100):
        ensure_talib()
        self.default_prob = default_prob
        self.window = window
        self.funcs: Dict[str, callable] = {
            fn: getattr(talib, fn)
            for fn in talib.get_function_groups().get("Pattern Recognition", [])
        }
        self.history: Dict[str, deque] = {fn: deque(maxlen=window) for fn in self.funcs}

    def update_prior(self, fn_name: str, correct: bool):
        """Append outcome (1/0) for a given pattern name."""
        if fn_name not in self.history:
            self.history[fn_name] = deque(maxlen=self.window)
        self.history[fn_name].append(1 if correct else 0)

    def get_prior(self, fn_name: str) -> float:
        hist = self.history.get(fn_name)
        if not hist:
            return self.default_prob
        return float(np.mean(hist))

    def detect(self, df: pd.DataFrame):
        """
        Detects all TA-Lib patterns on the latest closed bar.
        Returns a list of tuples: (fn_name, value, prob, rolling_hit_rate, sample_size)
        where 'prob' is the current prior used, and hit-rate/size are for logging.
        """
        if len(df) < 5:
            return []
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        signals = []
        for fn_name, fn in self.funcs.items():
            try:
                out = fn(o, h, l, c)
                val = int(out[-1])
            except Exception:
                continue
            if val != 0:
                hist = self.history.get(fn_name, deque(maxlen=self.window))
                hit_rate = float(np.mean(hist)) if len(hist) > 0 else self.default_prob
                prob = self.get_prior(fn_name)
                signals.append((fn_name, val, prob, hit_rate, len(hist)))
        return signals


# --------------------------
# ATR helper
# --------------------------

def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period + 1:
        return None
    atr_series = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
    atr = float(atr_series[-1])
    if np.isnan(atr):
        return None
    return atr


# --------------------------
# Predictor
# --------------------------

class Predictor:
    """
    Weighted vote by pattern priors + small momentum tilt; ATR scales confidence.
    """
    def predict_next(self, df: pd.DataFrame, signals, atr: Optional[float] = None):
        score = 0.0

        for fn_name, val, prob, _, _ in signals:
            # TA-Lib returns magnitude (e.g., 100/200). Use magnitude as a base weight.
            w = max(0.1, abs(val) / 100.0)
            if val > 0:
                score += w * (prob - 0.5)
            else:
                score -= w * (prob - 0.5)

        # Momentum tilt from last 3 closes
        closes = df['close'].tail(3).values
        if len(closes) >= 3:
            mom = np.sign(closes[-1] - closes[0]) * 0.05
            score += mom

        # ATR scaling
        if atr is not None and df['close'].iloc[-1] != 0:
            pct = atr / df['close'].iloc[-1]
            if pct > 0.01:       # >1% of price -> boost
                score *= 1.2
            elif pct < 0.002:    # <0.2% -> dampen
                score *= 0.8

        # Map score to direction/confidence
        conf = float(min(1.0, max(0.0, 0.5 + score)))
        if score > 0.02:
            direction = "bullish"
        elif score < -0.02:
            direction = "bearish"
        else:
            direction = "neutral"
        return {"direction": direction, "confidence": conf}


# --------------------------
# OHLC Builder
# --------------------------

class OHLCBuilder:
    def __init__(self, window: int = 300):
        self.window = window
        self.ohlc = collections.OrderedDict()

    def update_tick(self, minute_key: datetime, ltp: float):
        bar = self.ohlc.get(minute_key)
        if bar is None:
            self.ohlc[minute_key] = {"open": ltp, "high": ltp, "low": ltp, "close": ltp}
            while len(self.ohlc) > self.window:
                self.ohlc.popitem(last=False)
        else:
            bar["high"] = max(bar["high"], ltp)
            bar["low"] = min(bar["low"], ltp)
            bar["close"] = ltp

    def as_dataframe(self) -> pd.DataFrame:
        if not self.ohlc:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(self.ohlc).T.sort_index()
        df.index.name = "time"
        return df


# --------------------------
# Dhan WebSocket Client
# --------------------------

class DhanFeedClient:
    def __init__(self, client_id: str, access_token: str):
        if websocket is None:
            raise RuntimeError("Please install websocket-client")
        self.client_id = client_id
        self.access_token = access_token
        self.ws = None
        self.on_quote = None  # callback(exch, security_id, quote_dict)

    def _url(self) -> str:
        # v2 WS auth via query parameters
        return (f"wss://api-feed.dhan.co/?token={self.access_token}"
                f"&clientId={self.client_id}&version=2&authType=2")

    def connect(self):
        self.ws = websocket.WebSocketApp(
            self._url(),
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        t = threading.Thread(
            target=self.ws.run_forever,
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
            daemon=True
        )
        t.start()

    def subscribe_quotes(self, instruments: List[Tuple[int, int]]):
        req = {
            "RequestCode": REQ_SUB_QUOTE,
            "InstrumentList": [{"ExchangeSegment": ex, "SecurityId": sid} for ex, sid in instruments],
        }
        try:
            self.ws.send(json.dumps(req))
            logging.info("Subscribed to %d instrument(s) for Quote.", len(instruments))
        except Exception as e:
            logging.exception("Subscription send failed: %s", e)

    # ---- Callbacks ----

    def _on_open(self, ws):
        logging.info("WebSocket opened")

    def _on_error(self, ws, error):
        logging.error("WebSocket error: %s", error)

    def _on_close(self, ws, status_code, msg):
        logging.warning("WebSocket closed: %s %s", status_code, msg)

    def _on_message(self, ws, message):
        if isinstance(message, (bytes, bytearray)):
            self._handle_binary(message)
        else:
            # occasinal acks/heartbeats
            logging.debug("Text WS message: %s", message)

    def _handle_binary(self, payload: bytes):
        if len(payload) < HDR_SIZE:
            return
        code, msg_len, exch, security_id = struct.unpack_from(HDR_FMT, payload, 0)
        if len(payload) < HDR_SIZE + msg_len:
            return

        body = payload[HDR_SIZE:HDR_SIZE + msg_len]
        if code == RES_QUOTE and len(body) >= QUOTE_SIZE:
            (ltp, ltq, ltt, atp, vol, tot_sell, tot_buy,
             day_open, day_close, day_high, day_low) = struct.unpack_from(QUOTE_FMT, body, 0)
            quote = {
                "ltp": float(ltp),
                "ltt": int(ltt),
                "exchange_segment": int(exch),
                "security_id": int(security_id),
            }
            if self.on_quote:
                self.on_quote(exch, security_id, quote)
        elif code == RES_FEED_DISCONNECT:
            logging.warning("Feed disconnect packet received")
        else:
            # ignore other codes in this script
            pass


# --------------------------
# Telegram
# --------------------------

class Telegram:
    def __init__(self, token: str, chat_id: str):
        if requests is None:
            raise RuntimeError("requests is required for Telegram")
        self.token = token
        self.chat_id = chat_id

    def send_text(self, text: str) -> bool:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=10
            )
            return r.ok
        except Exception:
            logging.exception("Telegram send_text failed")
            return False

    def send_image(self, path: str, caption: str = "") -> bool:
        try:
            with open(path, "rb") as f:
                r = requests.post(
                    f"https://api.telegram.org/bot{self.token}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"photo": f},
                    timeout=20
                )
            return r.ok
        except Exception:
            logging.exception("Telegram send_image failed")
            return False


# --------------------------
# Chart export
# --------------------------

def export_chart(df: pd.DataFrame, signals, path: str):
    """
    Minimal dependency candlestick plot for the last ~30 bars.
    Highlights the latest closed bar; annotates first detected pattern name.
    """
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df_plot = df.tail(30).copy()
    o = df_plot['open'].values
    h = df_plot['high'].values
    l = df_plot['low'].values
    c = df_plot['close'].values
    x = np.arange(len(df_plot))

    for i in range(len(df_plot)):
        up = c[i] >= o[i]
        color = "green" if up else "red"
        # Wick
        ax.plot([x[i], x[i]], [l[i], h[i]], linewidth=1.0, color=color)
        # Body
        body_low = min(o[i], c[i])
        body_h = abs(c[i] - o[i]) if abs(c[i] - o[i]) > 1e-12 else 0.000001
        ax.add_patch(plt.Rectangle((x[i] - 0.3, body_low), 0.6, body_h, linewidth=0.0, color=color, alpha=0.8))

    # Highlight latest bar outline
    if len(df_plot) > 0:
        ax.add_patch(plt.Rectangle((x[-1] - 0.35, min(o[-1], c[-1])),
                                   0.7, abs(c[-1] - o[-1]) if abs(c[-1] - o[-1]) > 1e-12 else 0.000001,
                                   fill=False, linewidth=1.5, edgecolor="blue"))

    if signals:
        label = f"{signals[0][0]} ({'Bull' if signals[0][1]>0 else 'Bear'})"
        ax.annotate(label, (x[-1], c[-1]), xytext=(x[-1], c[-1] * 1.01),
                    arrowprops=dict(arrowstyle="->"))

    ax.set_title("Last ~30 candles (latest bar highlighted)")
    ax.set_xlim(x[0] - 1, x[-1] + 1)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------
# Bot (glues everything)
# --------------------------

class Bot:
    def __init__(self, args):
        self.args = args
        self.builder = OHLCBuilder(window=args.window)
        self.engine = PatternEngine(default_prob=args.default_prob, window=100)
        self.predictor = Predictor()
        self.tg = (Telegram(args.telegram_token, args.telegram_chat_id)
                   if args.telegram_token and args.telegram_chat_id else None)
        self.client = None if args.replay_csv else DhanFeedClient(args.client_id, args.access_token)
        if self.client:
            self.client.on_quote = self.on_quote

        # Parse instruments "SEG:SID" -> (segment_id, security_id)
        self.instruments: List[Tuple[int, int]] = []
        for token in args.instruments:
            seg, sid = token.split(":")
            seg = seg.strip().upper()
            if seg not in EXCHANGE_SEGMENT_MAP:
                raise ValueError(f"Unknown exchange segment: {seg}")
            self.instruments.append((EXCHANGE_SEGMENT_MAP[seg], int(sid)))

        self.last_minute: Optional[datetime] = None

        # For rolling prior updates:
        # Store the last prediction (based on signals at the time) so when the next bar
        # closes we can measure correctness and update per-pattern priors.
        self.prev_prediction = None  # dict with keys: close_idx, prev_close, pred_dir, signals

        # Ensure chart folder exists (even if we send to Telegram, we still save the file)
        self.chart_dir = os.path.abspath(self.args.chart_dir)
        os.makedirs(self.chart_dir, exist_ok=True)

    # ---- Live / Replay entrypoints ----

    def start(self):
        if self.client:
            self.client.connect()
            time.sleep(1.0)
            self.client.subscribe_quotes(self.instruments)
            logging.info("Live mode started; listening for data...")
            # keep main thread alive
            while True:
                time.sleep(1.0)
        else:
            self.replay_csv(self.args.replay_csv)

    def replay_csv(self, path: str):
        logging.info("Replay mode: %s", path)
        df = pd.read_csv(path)
        # Expect at least: time,open,high,low,close
        if 'time' not in df.columns:
            raise ValueError("CSV must have a 'time' column (ISO8601 or epoch seconds)")
        # Parse time column
        try:
            df['time'] = pd.to_datetime(df['time'], utc=True)
        except Exception:
            # Try epoch seconds
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

        for _, row in df.iterrows():
            minute_key = row['time'].to_pydatetime().replace(second=0, microsecond=0)
            # Use close as tick value; for replay we only have completed bars
            self.builder.update_tick(minute_key, float(row['close']))
            self._maybe_eval(minute_key)

        logging.info("Replay finished")

    # ---- Stream handlers ----

    def on_quote(self, exch: int, sid: int, q: Dict):
        minute_key = ts_to_minute(q["ltt"])
        self.builder.update_tick(minute_key, q["ltp"])
        self._maybe_eval(minute_key)

    def _maybe_eval(self, minute_key: datetime):
        """
        Called on every tick/replay row. When minute changes, the previous minute bar
        is considered closed -> run detection+prediction, and also update the
        rolling priors using the outcome of the *previous* prediction (if any).
        """
        if self.last_minute is None:
            self.last_minute = minute_key
            return

        if minute_key == self.last_minute:
            return  # still the same minute; keep building the bar

        # Minute rolled -> last closed bar is at index -1 now
        df = self.builder.as_dataframe()
        if len(df) < 2:
            self.last_minute = minute_key
            return

        # 1) Update priors from the previous prediction (if we had one)
        if self.prev_prediction is not None and len(df) >= 2:
            prev_close = df['close'].iloc[-2]  # close of bar we predicted from
            new_close = df['close'].iloc[-1]   # close of the bar that just closed
            move = new_close - prev_close
            actual_dir = "bullish" if move > 0 else "bearish" if move < 0 else "neutral"
            pred_dir = self.prev_prediction['pred_dir']
            correct = (actual_dir == pred_dir) if pred_dir != "neutral" else (actual_dir == "neutral")

            for fn_name, val, prob, _, _ in self.prev_prediction['signals']:
                # Update each involved pattern's rolling hit-rate
                self.engine.update_prior(fn_name, correct)

            logging.debug("Updated priors from previous prediction: "
                          "pred=%s actual=%s correct=%s", pred_dir, actual_dir, correct)

        # 2) Detect patterns on the latest closed bar and predict the next one
        signals = self.engine.detect(df)
        atr = compute_atr(df)
        pred = self.predictor.predict_next(df, signals, atr)

        # Log message with rolling accuracy per pattern
        last_close = df['close'].iloc[-1]
        msg_lines = [
            f"Close {last_close:.4f} at {self.last_minute.isoformat()} (UTC)",
            f"Next candle → {pred['direction'].upper()} (confidence {pred['confidence']:.2f})"
        ]
        if signals:
            parts = []
            for fn_name, val, prob, hit_rate, n_samples in signals:
                parts.append(f"{fn_name}({'Bull' if val>0 else 'Bear'})@{prob:.2f}[hit={hit_rate:.2f}/{n_samples}]")
            msg_lines.insert(1, "Patterns: " + ", ".join(parts))
        else:
            msg_lines.insert(1, "Patterns: None")
        msg = "\n".join(msg_lines)
        logging.info(msg)

        # 3) Chart export & Telegram send when any pattern fires
        chart_path = os.path.join(self.chart_dir, f"chart_{int(time.time())}.png")
        try:
            export_chart(df, signals, chart_path)
            if self.tg:
                self.tg.send_text(msg)
                self.tg.send_image(chart_path, "Pattern chart")
        except Exception:
            logging.exception("Chart export / Telegram send failed")

        # 4) Store current prediction so next minute we can score it
        self.prev_prediction = {
            "pred_dir": pred['direction'],
            "signals": signals,
        }

        # advance marker
        self.last_minute = minute_key


# --------------------------
# CLI
# --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dhan v2 Live Pattern Detector with rolling priors, ATR, charts, and replay.")
    p.add_argument("--client-id", default="", help="Dhan clientId (live mode)")
    p.add_argument("--access-token", default="", help="Dhan access token (live mode)")
    p.add_argument("--instruments", nargs="+", default=["IDX_I:13"], help="List like SEG:SECURITY_ID")
    p.add_argument("--window", type=int, default=300, help="Number of 1m bars to retain")
    p.add_argument("--default-prob", type=float, default=0.55, help="Fallback probability prior")
    p.add_argument("--telegram-token", default="", help="Telegram bot token")
    p.add_argument("--telegram-chat-id", default="", help="Telegram chat id")
    p.add_argument("--chart-dir", default="charts", help="Folder to save charts")
    p.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, ...)")
    p.add_argument("--replay-csv", default="", help="Path to OHLC CSV (time,open,high,low,close) for offline replay")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    if not args.replay_csv:
        # Live mode sanity checks
        if not args.client_id or not args.access_token:
            logging.error("Live mode requires --client-id and --access-token (or use --replay-csv for offline).")
            sys.exit(2)

    try:
        bot = Bot(args)
    except Exception as e:
        logging.exception("Initialization failed: %s", e)
        sys.exit(2)

    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nShutting down.")
    except Exception as e:
        logging.exception("Fatal error: %s", e)


if __name__ == "__main__":
    main()
