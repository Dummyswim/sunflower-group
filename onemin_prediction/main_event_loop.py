# main_event_loop.py
import asyncio
import base64
import json
import math
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
from contextlib import suppress

import numpy as np
import pandas as pd
import websockets

from core_handler import UnifiedWebSocketHandler as WSHandler
from feature_pipeline import FeaturePipeline, TA
from logging_setup import log_every
from model_pipeline import create_default_pipeline


logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))

# ========== FUTURES SIDECAR FEATURE INGEST ==========

_FUT_CACHE = {"mtime": None, "last_row": None, "prev_row": None}

def _read_latest_fut_features(path: str, spot_last_px: float) -> Dict[str, float]:
    """
    Read latest futures VWAP/CVD/volume features from sidecar CSV.
    Returns bounded, NaN-safe regime/orderflow proxies.
    """
    feats: Dict[str, float] = {}
    try:
        if not path or not os.path.exists(path):
            return feats

        mtime = os.path.getmtime(path)
        if _FUT_CACHE["mtime"] != mtime:
            df = pd.read_csv(path)
            if df is None or df.empty:
                return feats

            # Headerless sidecar fallback: assign expected columns when missing
            if "session_vwap" not in df.columns:
                expected = ["ts", "open", "high", "low", "close", "volume", "tick_count", "session_vwap", "cvd", "cum_volume"]
                cols = expected[: len(df.columns)]
                cols += [f"col_{i}" for i in range(len(df.columns) - len(cols))]
                df.columns = cols

            if "cum_volume" not in df.columns:
                if "volume" in df.columns:
                    df["cum_volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).cumsum()
                elif "tick_count" in df.columns:
                    df["cum_volume"] = pd.to_numeric(df["tick_count"], errors="coerce").fillna(0.0).cumsum()
                else:
                    df["cum_volume"] = 0.0
            if "cvd" not in df.columns:
                df["cvd"] = 0.0

            df = df.tail(2).copy()
            _FUT_CACHE["mtime"] = mtime
            _FUT_CACHE["prev_row"] = df.iloc[0].to_dict() if len(df) > 1 else None
            _FUT_CACHE["last_row"] = df.iloc[-1].to_dict()

        last = _FUT_CACHE["last_row"] or {}
        prev = _FUT_CACHE["prev_row"] or {}

        cur_vwap = float(last.get("session_vwap", 0.0) or 0.0)
        cur_cvd = float(last.get("cvd", 0.0) or 0.0)
        cur_vol = float(last.get("cum_volume", 0.0) or 0.0)

        prev_cvd = float(prev.get("cvd", cur_cvd) or cur_cvd)
        prev_vol = float(prev.get("cum_volume", cur_vol) or cur_vol)

        cvd_delta = cur_cvd - prev_cvd
        vol_delta = cur_vol - prev_vol

        # Bounded order-flow proxies
        cvd_norm = float(np.tanh(cvd_delta / max(1.0, cur_vol)))
        vol_norm = float(np.tanh(vol_delta / 10000.0))

        if cur_vwap > 0.0 and spot_last_px > 0.0:
            vwap_dev = (spot_last_px - cur_vwap) / max(1e-9, cur_vwap)
        else:
            vwap_dev = 0.0
        vwap_dev = float(np.clip(vwap_dev, -0.01, 0.01))

        feats.update({
            "fut_session_vwap": cur_vwap,
            "fut_vwap_dev": vwap_dev,
            "fut_cvd_delta": cvd_norm,
            "fut_vol_delta": vol_norm,
        })
    except Exception as e:
        logger.debug(f"[FUT] read_latest_fut_features failed: {e}", exc_info=True)
    return feats


def _compute_vol_features(candle_df: pd.DataFrame) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    try:
        if candle_df is None or candle_df.empty:
            return feats
        df = candle_df.tail(30).copy()

        hi = df["high"].astype(float).values
        lo = df["low"].astype(float).values
        cl = df["close"].astype(float).values

        rng = np.maximum(0.0, hi - lo)
        feats["atr_1t"] = float(np.mean(rng[-5:])) if len(rng) >= 5 else float(np.mean(rng))
        feats["atr_3t"] = float(np.mean(rng[-15:])) if len(rng) >= 15 else feats["atr_1t"]

        if len(cl) >= 10:
            rets = np.diff(cl) / np.maximum(1e-9, cl[:-1])
            feats["rv_10"] = float(np.std(rets[-10:]))
        else:
            feats["rv_10"] = 0.0
    except Exception as e:
        logger.debug(f"[VOL] compute_vol_features failed: {e}", exc_info=True)
    return feats


def _time_of_day_features(ts: datetime) -> Dict[str, float]:
    """
    Encode minutes since 09:15 IST as sin/cos.
    """
    feats: Dict[str, float] = {}
    try:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)

        open_ts = ts.replace(hour=9, minute=15, second=0, microsecond=0)
        mins = max(0.0, (ts - open_ts).total_seconds() / 60.0)
        ang = 2.0 * np.pi * (mins / 375.0)  # ~6.25h session
        feats["tod_sin"] = float(np.sin(ang))
        feats["tod_cos"] = float(np.cos(ang))
    except Exception:
        feats["tod_sin"] = 0.0
        feats["tod_cos"] = 0.0
    return feats


# ========== ADAPTIVE CONFIDENCE TUNER ==========

class RollingConfidenceTuner:
    """
    Tracks rolling hit-rate & Brier to adjust qmin.

    dir_true:
      +1  -> BUY label
      -1  -> SELL label
       0  -> FLAT (no directional move; we expect p≈0.5)
    """

    def __init__(self, window: int = 80) -> None:
        self.window = max(10, int(window))
        self._hits: Deque[float] = deque(maxlen=self.window)
        self._briers: Deque[float] = deque(maxlen=self.window)

    def update(self, dir_true: int, buy_prob: float) -> None:
        """
        dir_true: +1 (BUY), -1 (SELL), 0 (FLAT)
        buy_prob: model's BUY probability at decision time.
        """
        try:
            p = float(buy_prob)
        except Exception:
            p = 0.5
        p = max(0.0, min(1.0, p))

        dir_pred = 1 if p > 0.5 else (-1 if p < 0.5 else 0)

        if dir_true in (1, -1):
            hit = 1.0 if dir_pred == dir_true else 0.0
            target = 1.0 if dir_true == 1 else 0.0
            brier = (p - target) ** 2
        elif dir_true == 0:
            conf = abs(p - 0.5)
            hit = 1.0 if conf < 0.10 else 0.0
            brier = (p - 0.5) ** 2
        else:
            return

        self._hits.append(hit)
        self._briers.append(brier)

    def qmin_delta(self) -> float:
        """
        Suggest a small additive adjustment to qmin:
        - If hit-rate is clearly good (≥0.60) → slightly lower qmin.
        - If hit-rate is clearly poor (≤0.40) → slightly raise qmin.
        - Otherwise → no change.

        Returns 0.0 until we have enough samples.
        """
        n = len(self._hits)
        min_active = max(5, int(self.window * 0.5))
        if n < min_active:
            return 0.0

        hit_rate = float(sum(self._hits) / n) if n else 0.0
        brier = float(sum(self._briers) / n) if n else 0.0

        delta = 0.0
        if hit_rate >= 0.60:
            delta = -0.01
        elif hit_rate <= 0.40:
            delta = +0.01

        logger.info(
            "[CONF] tuner snapshot: n=%d hit=%.3f brier=%.3f delta=%+.4f",
            n, hit_rate, brier, delta,
        )
        return delta

    def snapshot(self) -> Dict[str, float]:
        return {
            "hit_rate": float(np.mean(self._hits)) if self._hits else 0.0,
            "brier": float(np.mean(self._briers)) if self._briers else 0.0,
            "n": float(len(self._hits)),
        }


def _load_q_model(path: str):
    """Deprecated: kept for compatibility."""
    return None


class QMetaModel2Min:
    """
    Lightweight loader for offline-trained Q logistic model.

    Expects JSON with keys:
      - feature_names: list[str]
      - coef: [ [w1, w2, ...] ] or [w1, w2, ...]
      - intercept: float
    """

    def __init__(self, feature_names: Sequence[str], coef: Sequence[float], intercept: float):
        self.feature_names = list(feature_names)
        self.coef = [float(c) for c in coef]
        self.intercept = float(intercept)

    @classmethod
    def from_env(cls, logger: logging.Logger) -> Optional["QMetaModel2Min"]:
        path = os.getenv("Q_MODEL_2MIN_PATH")
        if not path:
            logger.info("[Q-MODEL] Q_MODEL_2MIN_PATH not set -> Q gating disabled")
            return None
        try:
            with open(path, "r") as f:
                blob = json.load(f)
            feat_names = blob.get("feature_names") or blob.get("features")
            coef = blob.get("coef")
            if isinstance(coef, list) and coef and isinstance(coef[0], list):
                coef = coef[0]
            intercept = blob.get("intercept", 0.0)
            if not feat_names or coef is None:
                logger.warning("[Q-MODEL] Invalid format in %s (missing feature_names/coef)", path)
                return None
            model = cls(feat_names, coef, intercept)
            logger.info("[Q-MODEL] Loaded Q model from %s (n_features=%d)", path, len(model.feature_names))
            logger.debug("[Q-MODEL] feature_names=%s", model.feature_names)
            return model
        except FileNotFoundError:
            logger.warning("[Q-MODEL] File not found: %s -> Q gating disabled", path)
        except Exception as exc:
            logger.exception("[Q-MODEL] Failed to load model from %s: %s", path, exc)
        return None

    def predict_q_hat(self, p_buy: float, feature_source: Dict[str, float]) -> float:
        """
        Compute P(correct) using logistic regression.

        feature_source: dict with raw feature values (e.g., cvd_divergence, wick_extreme_*).
        """
        z = self.intercept
        for name, w in zip(self.feature_names, self.coef):
            if name == "p_buy":
                val = float(p_buy)
            else:
                v = feature_source.get(name, 0.0)
                try:
                    val = float(v)
                except Exception:
                    val = 0.0
            z += w * val
        # logistic
        try:
            q_hat = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            q_hat = 1.0 if z > 0 else 0.0
        # We keep semantics: Q ∈ [0, 0.5] (0.5 = best)
        q_clamped = max(0.0, min(0.5, float(q_hat)))
        return q_clamped


# ========== HEARTBEAT / WATCHDOG / WS UTILITIES ==========

async def _websocket_heartbeat(name: str, ws_handler: WSHandler, interval_sec: int = 30):
    while True:
        try:
            ticks = getattr(ws_handler, "tick_count", 0)
            last_ts = getattr(ws_handler, "last_packet_time", None)
            if last_ts and hasattr(last_ts, "timestamp"):
                age = max(0.0, time.time() - last_ts.timestamp())
                logger.info("[%s] Websocket is active. ticks=%d, last_packet_age=%.1fs", name, ticks, age)
            else:
                logger.info("[%s] Websocket is active. ticks=%d, awaiting first packet...", name, ticks)
        except asyncio.CancelledError:
            logger.info(f"[{name}] Heartbeat cancelled")
            break
        except Exception as e:
            logger.debug(f"[{name}] Heartbeat error (ignored): {e}")
        finally:
            await asyncio.sleep(interval_sec)

def _build_dhan_ws_url(cfg: Any) -> Optional[str]:
    try:
        tok_b64 = getattr(cfg, "dhan_access_token_b64", "") or ""
        cid_b64 = getattr(cfg, "dhan_client_id_b64", "") or ""
        if not tok_b64 or not cid_b64:
            return None
        access_token = base64.b64decode(tok_b64).decode("utf-8")
        client_id = base64.b64decode(cid_b64).decode("utf-8")
        return ("wss://api-feed.dhan.co"
                f"?version=2&token={access_token}&clientId={client_id}&authType=2")
    except Exception as e:
        logger.error(f"Failed to build Dhan WS URL: {e}")
        return None

def _subscription_payload(cfg: Any) -> Dict[str, Any]:
    return {
        "RequestCode": 15,
        "InstrumentCount": 1,
        "InstrumentList": [{
            "ExchangeSegment": getattr(cfg, "nifty_exchange_segment", "IDX_I"),
            "SecurityId": str(getattr(cfg, "nifty_security_id", "")),
        }]
    }

async def _data_stall_watchdog(name: str, ws_handler: WSHandler, resubscribe_cb, reconnect_cb, stall_secs: int, reconnect_secs: int):
    did_resubscribe = False
    last_sub_time: Optional[datetime] = None

    def set_last_sub_time(ts: datetime):
        nonlocal last_sub_time
        last_sub_time = ts

    setattr(reconnect_cb, "_set_last_sub_time", set_last_sub_time)

    while True:
        try:
            await asyncio.sleep(1)
            now = datetime.now(IST)
            last_pkt = getattr(ws_handler, "last_packet_time", None)
            if last_pkt is None:
                if last_sub_time:
                    since_sub = (now - last_sub_time).total_seconds()
                    logger.info(f"[{name}] Watchdog: {since_sub:.1f}s since subscribe, no packets yet")
                    if since_sub >= stall_secs and not did_resubscribe:
                        logger.warning(f"[{name}] No data for {stall_secs}s after subscribe — re-subscribing")
                        try:
                            await resubscribe_cb()
                            did_resubscribe = True
                            logger.info(f"[{name}] Resubscribe issued at {now.strftime('%H:%M:%S')}")
                        except Exception as e:
                            logger.error(f"[{name}] Resubscribe failed: {e}")
                    if since_sub >= reconnect_secs:
                        logger.warning(f"[{name}] No data for {reconnect_secs}s — reconnecting")
                        await reconnect_cb()
                        break
                continue
            else:
                did_resubscribe = False
        except asyncio.CancelledError:
            logger.debug(f"[{name}] Watchdog cancelled")
            break
        except Exception as e:
            logger.error(f"[{name}] Watchdog error: {e}", exc_info=True)
            await asyncio.sleep(2)

async def _ws_connect_and_stream(name: str, cfg: Any, ws_handler: WSHandler, stop_event: asyncio.Event):
    backoff_base = int(getattr(cfg, "reconnect_delay_base", 2)) or 2
    max_attempts_cfg = getattr(cfg, "max_reconnect_attempts", 5)
    try:
        max_attempts = int(max_attempts_cfg)
    except Exception:
        max_attempts = 5
    infinite = max_attempts <= 0

    while not stop_event.is_set():
        ws_url = _build_dhan_ws_url(cfg)
        if not ws_url:
            logger.critical(f"[{name}] Missing or invalid credentials; cannot build WS URL")
            return

        attempt = 0
        while (infinite or attempt < max_attempts) and not stop_event.is_set():
            attempt += 1
            backoff = min(backoff_base * (2 ** (attempt - 1)), 60)
            try:
                logger.info(f"[{name}] Connecting to wss://api-feed.dhan.co?[masked] (attempt {attempt}/{max_attempts if not infinite else '∞'})")
                ping_interval = int(getattr(cfg, "ws_ping_interval", 30)) or 30
                ping_timeout = int(getattr(cfg, "ws_ping_timeout", 10)) or 10
                async with websockets.connect(
                    ws_url,
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    max_size=10 * 1024 * 1024,
                    compression=None,
                    open_timeout=30,
                    close_timeout=10
                ) as ws:
                    logger.info(f"[{name}] WebSocket connected")
                    sub = _subscription_payload(cfg)
                    last_sub_time = datetime.now(IST)
                    await ws.send(json.dumps(sub))
                    logger.info(f"[{name}] Subscription sent at {last_sub_time.strftime('%H:%M:%S')}")

                    stall_secs = int(getattr(cfg, "data_stall_seconds", 15)) or 15
                    reconn_secs = int(getattr(cfg, "data_stall_reconnect_seconds", 30)) or 30

                    async def resubscribe():
                        nonlocal last_sub_time
                        await ws.send(json.dumps(sub))
                        last_sub_time = datetime.now(IST)
                        logger.info(f"[{name}] Resubscription sent at {last_sub_time.strftime('%H:%M:%S')}")

                    async def reconnect():
                        try:
                            await ws.close()
                        except Exception:
                            pass

                    watchdog_task = asyncio.create_task(
                        _data_stall_watchdog(name, ws_handler, resubscribe, reconnect, stall_secs, reconn_secs)
                    )
                    setter = getattr(reconnect, "_set_last_sub_time", None)
                    if callable(setter):
                        setter(last_sub_time)

                    try:
                        async for message in ws:
                            if stop_event.is_set():
                                break
                            if isinstance(message, bytes):
                                tick_data = None
                                try:
                                    if len(message) == 16 and message and message[0] == ws_handler.TICKER_PACKET:
                                        if hasattr(ws_handler, "_parse_ticker_packet"):
                                            tick_data = ws_handler._parse_ticker_packet(message)
                                except Exception as e:
                                    logger.error(f"[{name}] Parse error: {e}", exc_info=True)
                                if tick_data:
                                    await ws_handler._process_tick(tick_data)
                            else:
                                try:
                                    data = json.loads(message)
                                    code = data.get("ResponseCode")
                                    if code:
                                        logger.info(f"[{name}] Control: code={code} msg={data.get('ResponseMessage', '')}")
                                except Exception:
                                    logger.debug(f"[{name}] Text message: {str(message)[:200]}")
                            if ws_handler.last_packet_time is None and callable(setter):
                                setter(last_sub_time)
                    except asyncio.CancelledError:
                        logger.info(f"[{name}] Message loop cancelled")
                        raise
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"[{name}] WS connection closed: {e}")
                    except Exception as e:
                        logger.error(f"[{name}] Fatal message loop error: {e}", exc_info=True)
                    finally:
                        try:
                            watchdog_task.cancel()
                            await asyncio.wait_for(watchdog_task, timeout=3.0)
                        except Exception:
                            pass

                if stop_event.is_set():
                    return
                logger.warning(f"[{name}] Message loop ended; reconnecting after {backoff}s")
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                logger.info(f"[{name}] Connector cancelled")
                return
            except Exception as e:
                level = logger.warning if "no close frame" in str(e).lower() else logger.error
                level(f"[{name}] Connect attempt failed: {e}")
                if infinite or attempt < max_attempts:
                    logger.info(f"[{name}] Retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                else:
                    logger.critical(f"[{name}] Failed to establish WS after {max_attempts} attempts")
                    return
        if stop_event.is_set():
            return
        await asyncio.sleep(1)


# ========== HELPERS ==========

def _extract_best_and_mid_from_tick(tick: Dict[str, Any]) -> Tuple[float, float]:
    try:
        ltp = float(tick.get('ltp', 0.0))
    except Exception:
        ltp = 0.0
    bid = ask = ltp
    try:
        md = tick.get('market_depth')
        if isinstance(md, list) and md:
            best = md[0]
            bid = float(best.get('bid_price', bid))
            ask = float(best.get('ask_price', ask))
    except Exception:
        pass
    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else ltp
    best_price = ltp if ltp > 0 else mid
    return best_price, mid


# ========== MAIN LOOP (PREDICTIVE-ONLY, PRE-CLOSE TRIGGER) ==========

async def main_loop(config, cnn_lstm, xgb, train_features, token_b64, chat_id, neutral_model=None):
    logger.info("=" * 60)
    logger.info("STARTING PROBABILITIES-ONLY MAIN EVENT LOOP")
    logger.info("=" * 60)

    # Normalize connections (single connection default)
    connections: List[Tuple[str, Any]] = []
    try:
        if hasattr(config, "connections") and isinstance(config.connections, dict):
            for name, cfg in config.connections.items():
                connections.append((str(name), cfg))
        elif hasattr(config, "connections") and hasattr(config, "__iter__"):
            for i, cfg in enumerate(config.connections, start=1):
                sec = getattr(cfg, "nifty_security_id", getattr(cfg, "security_id", "NA"))
                seg = getattr(cfg, "nifty_exchange_segment", getattr(cfg, "exchange_segment", "NA"))
                connections.append((f"conn{i}:{seg}:{sec}", cfg))
        else:
            sec = getattr(config, "nifty_security_id", getattr(config, "security_id", "NA"))
            seg = getattr(config, "nifty_exchange_segment", getattr(config, "exchange_segment", "NA"))
            connections.append((f"primary:{seg}:{sec}", config))
    except Exception:
        connections.append(("primary", config))

    logger.info(f"Total connections configured: {len(connections)}")

    # Shared components
    feat_pipe = FeaturePipeline(train_features=train_features)
    model_pipe = create_default_pipeline(cnn_lstm, xgb, neutral_model=neutral_model)
    try:
        model_pipe.replace_models(xgb=xgb)
        logger.info("Pipeline schema aligned to booster at startup (if embedded)")
    except Exception as e:
        logger.warning(f"Pipeline schema alignment skipped at startup: {e}")
    logger.info("Global components initialized successfully")
    q_model_2min = QMetaModel2Min.from_env(logger)

    # Paths
    feature_log_path = getattr(config, 'feature_log_path', 'feature_log.csv')
    logger.info(f"Feature log path: {feature_log_path}")

    # Background online trainer (kept)
    trainer_task = None
    try:
        from online_trainer import background_trainer_loop
        xgb_out = os.getenv("XGB_PATH", "trained_models/production/xgb_model.json")
        neutral_out = os.getenv("NEUTRAL_PATH", "trained_models/production/neutral_model.pkl")
        Path(Path(xgb_out).parent or ".").mkdir(parents=True, exist_ok=True)
        Path(Path(neutral_out).parent or ".").mkdir(parents=True, exist_ok=True)
        trainer_task = asyncio.create_task(background_trainer_loop(
            feature_log_path=feature_log_path,
            xgb_out_path=xgb_out,
            neutral_out_path=neutral_out,
            pipeline_ref=model_pipe,
            interval_sec=int(getattr(config, "trainer_interval_sec", 300)) if hasattr(config, "trainer_interval_sec") else 300,
            min_rows=int(getattr(config, "trainer_min_rows", 100)) if hasattr(config, "trainer_min_rows") else 100
        ))
        logger.info("Online trainer task started")
    except Exception as e:
        logger.warning(f"Online trainer not started: {e}")

    # Background Platt calibrator (kept)
    calib_task = None
    try:
        from calibrator import background_calibrator_loop
        calib_out = os.getenv("CALIB_PATH", "trained_models/production/platt_calibration.json")
        calib_task = asyncio.create_task(background_calibrator_loop(
            feature_log_path=feature_log_path,
            calib_out_path=calib_out,
            pipeline_ref=model_pipe,
            interval_sec=int(getattr(config, "calib_interval_sec", 1200)) if hasattr(config, "calib_interval_sec") else 1200,
            min_dir_rows=int(getattr(config, "calib_min_rows", 300)) if hasattr(config, "calib_min_rows") else 300
        ))
        logger.info(f"[CALIB] Calibrator task started → {calib_out}")
    except Exception as e:
        logger.warning(f"[CALIB] Calibrator not started: {e}")

    # Drift baseline refresher (optional)
    async def _maybe_refresh_drift_baseline():
        try:
            path = Path(feature_log_path)
            if not path.exists():
                return
            df = pd.read_csv(path, header=None,
                             names=["ts","decision","label","buy_prob","alpha","tradeable","is_flat","ticks","rest"],
                             on_bad_lines='skip')
            keep = ["ema_8","ema_21","spread","micro_slope","micro_imbalance","mean_drift_pct","last_zscore","last_price"]
            baselines = {k: [] for k in keep}
            sample_n = 600
            for row in df.tail(sample_n).itertuples(index=False):
                try:
                    toks = str(row.rest).split(",")
                    for t in toks:
                        if "=" not in t:
                            continue
                        k, v = t.split("=", 1)
                        if k in baselines:
                            baselines[k].append(float(v))
                except Exception:
                    continue
            baselines = {k: v for k, v in baselines.items() if v}
            if baselines:
                feat_pipe.train_features = baselines
                logger.info(f"[DRIFT] Baseline refreshed from feature_log: keys={list(baselines.keys())} samples~{min(len(next(iter(baselines.values()))), sample_n)}")
        except Exception as e:
            logger.debug(f"[DRIFT] Baseline refresh failed (ignored): {e}")

    async def drift_baseline_refresh_loop(interval_sec: int = 600):
        logger.info(f"[DRIFT] Baseline refresh loop started (every {interval_sec}s)")
        while True:
            try:
                await _maybe_refresh_drift_baseline()
            except asyncio.CancelledError:
                logger.info("[DRIFT] Baseline refresh loop cancelled")
                break
            except Exception as e:
                logger.debug(f"[DRIFT] Periodic refresh failed (ignored): {e}")
            await asyncio.sleep(interval_sec)

    baseline_task = asyncio.create_task(drift_baseline_refresh_loop(600))

    # Per-connection tasks and state
    stop_events: Dict[str, asyncio.Event] = {}
    ws_tasks: List[Tuple[str, WSHandler, asyncio.Task]] = []
    hb_tasks: List[Tuple[str, asyncio.Task]] = []

    for name, cfg in connections:
        try:
            ws_handler = WSHandler(cfg)

            ob_ring = deque(maxlen=10)
            staged_map: Dict[datetime, Dict[str, Any]] = {}   # features staged per reference candle start (2-min horizon)
            tuner = RollingConfidenceTuner(
                window=int(os.getenv("ROLLING_CONF_WINDOW", "50") or "50"),
            )

            async def _on_tick_cb(tick: Dict[str, Any]):
                try:
                    best_px, mid_px = _extract_best_and_mid_from_tick(tick)
                    ob_snapshot = {
                        'bid_price': best_px if best_px <= mid_px else mid_px,
                        'ask_price': best_px if best_px >= mid_px else mid_px
                    }
                    ob_ring.append(ob_snapshot)
                except Exception as e:
                    logger.debug(f"[{name}] on_tick callback error (ignored): {e}")

            # ---------- PRE-CLOSE: STAGE FEATURES AT CURRENT CANDLE START (2-MIN HORIZON) ----------
            async def _on_preclose_cb(preview_df: pd.DataFrame, full_df: pd.DataFrame):
                try:
                    if preview_df is None or preview_df.empty:
                        return

                    # Current bucket start time t (candle being closed now)
                    current_bucket_start = preview_df.index[-1]

                    interval_sec = int(getattr(cfg, 'candle_interval_seconds', 60))
                    interval_min = max(1, interval_sec // 60)
                    horizon_min = 2 * interval_min  # 2-minute target horizon

                    # Reference time for label & training row
                    ref_start = current_bucket_start
                    # Target close candle start (for logging only)
                    target_start = current_bucket_start + timedelta(minutes=horizon_min)

                    # Gather prices for TA/indicator scaling
                    try:
                        prices = ws_handler.get_prices(last_n=200) if hasattr(ws_handler, 'get_prices') else []
                    except Exception:
                        prices = []
                    if not prices or len(prices) < 2:
                        return

                    # TA/EMA
                    ema_feats = FeaturePipeline.compute_emas(prices)
                    try:
                        ta = TA.compute_ta_bundle(prices)
                    except Exception:
                        ta = {}

                    # Recent candles for MTF patterns and SR bundle
                    safe_df = full_df.tail(500) if isinstance(full_df, pd.DataFrame) and not full_df.empty else pd.DataFrame()

                    tf_list = getattr(cfg, "pattern_timeframes", ["1T", "3T", "5T"])
                    mtf = FeaturePipeline.compute_mtf_pattern_consensus(
                        candle_df=safe_df,
                        timeframes=tf_list,
                        rvol_window=int(getattr(cfg, 'pattern_rvol_window', 5)),
                        rvol_thresh=float(getattr(cfg, 'pattern_rvol_threshold', 1.2)),
                        min_winrate=float(getattr(cfg, 'pattern_min_winrate', 0.55))
                    ) or {}
                    pattern_features = FeaturePipeline.compute_candlestick_patterns(
                        candles=safe_df.tail(max(3, int(getattr(cfg, 'pattern_rvol_window', 5)))),
                        rvol_window=int(getattr(cfg, 'pattern_rvol_window', 5)),
                        rvol_thresh=float(getattr(cfg, 'pattern_rvol_threshold', 1.2)),
                        min_winrate=float(getattr(cfg, 'pattern_min_winrate', 0.55))
                    ) or {}
                    sr = FeaturePipeline.compute_sr_features(safe_df) if isinstance(safe_df, pd.DataFrame) and not safe_df.empty else {
                        "sr_1T_hi_dist": 0.0, "sr_1T_lo_dist": 0.0,
                        "sr_3T_hi_dist": 0.0, "sr_3T_lo_dist": 0.0,
                        "sr_5T_hi_dist": 0.0, "sr_5T_lo_dist": 0.0,
                        "sr_breakout_up": 0.0, "sr_breakout_dn": 0.0
                    }

                    # Order-flow and microstructure
                    try:
                        order_books = list(ob_ring)
                    except Exception:
                        order_books = []
                    ofd = FeaturePipeline.order_flow_dynamics(order_books)

                    last_px = float(prices[-1]) if prices else 0.0
                    try:
                        micro_ctx = ws_handler.get_micro_features()
                        px_arr = np.asarray(prices[-64:], dtype=float)
                        last_px = float(px_arr[-1]) if px_arr.size else last_px
                        try:
                            vol_short = float(micro_ctx.get('std_dltp_short', 0.0))
                            denom = max(1e-6, vol_short)
                            micro_slope_normed = float(micro_ctx.get('slope', 0.0)) / denom
                        except Exception:
                            micro_slope_normed = float(micro_ctx.get('slope', 0.0))
                        if float(micro_ctx.get('price_range_tightness', 0.0)) >= 0.995:
                            micro_slope_normed = float(np.tanh(micro_slope_normed))
                    except Exception:
                        micro_ctx = {}
                        micro_slope_normed = 0.0

                    indicator_features = {
                        'ema_trend': 1.0 if float(ema_feats.get('ema_8', 0.0)) > float(ema_feats.get('ema_21', 0.0))
                                     else (-1.0 if float(ema_feats.get('ema_8', 0.0)) < float(ema_feats.get('ema_21', 0.0)) else 0.0),
                        'micro_slope': micro_slope_normed,
                        'imbalance': float(micro_ctx.get('imbalance', 0.0)),
                        'mean_drift': float(micro_ctx.get('mean_drift_pct', 0.0)) / 100.0,
                        'std_dltp_short': float(micro_ctx.get('std_dltp_short', 0.0)),
                        'price_range_tightness': float(micro_ctx.get('price_range_tightness', 0.0)),
                    }
                    try:
                        indicator_score = model_pipe.compute_indicator_score(indicator_features)
                        logger.debug(f"[{name}] Indicator score: {indicator_score:.3f}")
                    except Exception:
                        indicator_score = 0.0

                    # Normalization scale
                    try:
                        px = np.array(prices[-64:], dtype=float)
                        scale = float(np.std(np.diff(px))) if px.size >= 3 else 1.0
                        scale = max(1e-6, scale)
                    except Exception:
                        scale = 1.0

                    # --- NEW: vol/regime/time-of-day features ---
                    vol_feats = _compute_vol_features(safe_df)
                    tod_ts = current_bucket_start if isinstance(current_bucket_start, datetime) else datetime.now(IST)
                    tod_feats = _time_of_day_features(tod_ts)

                    # --- NEW: futures sidecar features ---
                    fut_path = os.getenv("FUT_SIDECAR_PATH", "trained_models/production/fut_candles_vwap_cvd.csv")
                    fut_feats = _read_latest_fut_features(fut_path, spot_last_px=last_px)

                    if fut_feats:
                        logger.debug(f"[FUT] injected futures features: {fut_feats}")

                    # last_zscore
                    try:
                        last_zscore = 0.0
                        if len(prices) >= 2:
                            px_arr64 = np.array(prices[-64:], dtype=float)
                            if px_arr64.size >= 2:
                                px_last = float(px_arr64[-1])
                                px_mean32 = float(np.mean(px_arr64[-32:])) if px_arr64.size >= 32 else float(np.mean(px_arr64))
                                px_std32 = float(np.std(px_arr64[-32:])) if px_arr64.size >= 32 else float(np.std(px_arr64))
                                last_zscore = (px_last - px_mean32) / max(1e-9, px_std32)
                    except Exception:
                        last_zscore = 0.0

                    # --- Reversal / regime flags (wick extremes, VWAP reversion, CVD divergence)
                    try:
                        if isinstance(safe_df, pd.DataFrame) and not safe_df.empty:
                            last_candle = safe_df.iloc[-1]
                            prev_candle = safe_df.iloc[-2] if len(safe_df) >= 2 else last_candle
                        else:
                            # Build a minimal struct for wick calc from prices
                            if prices and len(prices) >= 2:
                                last_candle = {"open": prices[-2], "high": prices[-1], "low": prices[-1], "close": prices[-1]}
                                prev_candle = {"open": prices[-3] if len(prices) >= 3 else prices[-2], "high": prices[-2], "low": prices[-2], "close": prices[-2]}
                            else:
                                last_candle = None
                                prev_candle = None
                    except Exception:
                        last_candle = None
                        prev_candle = None

                    try:
                        wick_up, wick_down = FeaturePipeline._compute_wick_extremes(last_candle) if last_candle is not None else (0.0, 0.0)
                    except Exception:
                        wick_up, wick_down = 0.0, 0.0

                    # VWAP reversion flag – prefer futures session vwap if present
                    vwap_val = None
                    if "fut_session_vwap" in fut_feats:
                        try:
                            vwap_val = float(fut_feats.get("fut_session_vwap", None))
                        except Exception:
                            vwap_val = None

                    try:
                        px_hist_df = safe_df if (isinstance(safe_df, pd.DataFrame) and not safe_df.empty) else None
                        vwap_rev = FeaturePipeline._compute_vwap_reversion_flag(px_hist_df, vwap_val) if px_hist_df is not None else 0.0
                    except Exception:
                        vwap_rev = 0.0

                    # CVD divergence using futures sidecar delta if available
                    try:
                        fut_cvd_delta = fut_feats.get("fut_cvd_delta", None)
                        last_close_val = float(prices[-1]) if prices else 0.0
                        prev_close_val = float(prices[-2]) if len(prices) >= 2 else last_close_val
                        px_change = last_close_val - prev_close_val
                        cvd_div = FeaturePipeline._compute_cvd_divergence(px_change, fut_cvd_delta)
                    except Exception:
                        cvd_div = 0.0

                    rev_cross_feats: Dict[str, float] = FeaturePipeline.compute_reversal_cross_features(
                        safe_df.tail(20) if isinstance(safe_df, pd.DataFrame) else pd.DataFrame(),
                        {
                            "wick_extreme_up": float(wick_up),
                            "wick_extreme_down": float(wick_down),
                            "cvd_divergence": float(cvd_div),
                            "vwap_reversion_flag": float(vwap_rev),
                        },
                    )

                    ta_feats = ta or {}
                    pat_feats = pattern_features or {}
                    mtf_feats = mtf or {}

                    features_raw = {
                        **ema_feats,
                        **ta_feats,
                        **pat_feats,
                        **mtf_feats,
                        **sr,
                        **indicator_features,
                        **ofd,
                        **vol_feats,
                        **tod_feats,
                        **fut_feats,
                        'micro_slope': micro_slope_normed,
                        'micro_imbalance': float(micro_ctx.get('imbalance', 0.0)),
                        'mean_drift_pct': float(micro_ctx.get('mean_drift_pct', 0.0)),
                        'last_price': float(last_px),
                        'last_zscore': float(last_zscore),
                        'std_dltp_short': float(micro_ctx.get('std_dltp_short', 0.0)),
                        'price_range_tightness': float(micro_ctx.get('price_range_tightness', 0.0)),
                        "indicator_score": float(indicator_score),
                        # Reversal / regime flags
                        "wick_extreme_up": float(wick_up),
                        "wick_extreme_down": float(wick_down),
                        "vwap_reversion_flag": float(vwap_rev),
                        "cvd_divergence": float(cvd_div),
                    }
                    features_raw.update(rev_cross_feats)
                    features = FeaturePipeline.normalize_features(features_raw, scale=scale)

                    # Live tensor
                    try:
                        live_tensor = (ws_handler.get_live_tensor() if hasattr(ws_handler, 'get_live_tensor') else None)
                        if live_tensor is None:
                            arr = np.array(prices[-64:], dtype=float)
                            if arr.size == 0:
                                live_tensor = np.zeros((1, 64, 1), dtype=float)
                            else:
                                arr = (arr - arr.mean()) / max(1e-9, arr.std())
                                if arr.size < 64:
                                    pad = np.zeros(64 - arr.size, dtype=float)
                                    arr = np.concatenate([pad, arr])
                                else:
                                    arr = arr[-64:]
                                live_tensor = arr.reshape(1, 64, 1)
                    except Exception:
                        live_tensor = np.zeros((1, 64, 1), dtype=float)

                    # Predict
                    feat_keys_sorted = sorted(features.keys())
                    engineered_features = [features[k] for k in feat_keys_sorted]
                    signal_probs, neutral_prob = model_pipe.predict(
                        live_tensor=live_tensor,
                        engineered_features=engineered_features,
                        indicator_score=indicator_score,
                        pattern_prob_adjustment=float(pattern_features.get('probability_adjustment', 0.0)) if pattern_features else 0.0,
                        engineered_feature_names=feat_keys_sorted,
                        mtf_consensus=float(mtf.get("mtf_consensus", 0.0)) if mtf else 0.0,
                    )
                    buy_prob = float(signal_probs[0][1])

                    # --- Neutral bump in extreme reversal contexts (weak model only) ---
                    try:
                        crosses = [
                            features_raw.get("rev_cross_upper_wick_cvd", 0.0),
                            features_raw.get("rev_cross_upper_wick_vwap", 0.0),
                            features_raw.get("rev_cross_lower_wick_cvd", 0.0),
                            features_raw.get("rev_cross_lower_wick_vwap", 0.0),
                        ]
                        has_reversal_cross = any(float(c or 0.0) > 0.5 for c in crosses)
                    except Exception:
                        has_reversal_cross = False

                    if getattr(model_pipe, "model_quality", "weak") == "weak" and has_reversal_cross:
                        try:
                            old_neu = float(neutral_prob) if neutral_prob is not None else None
                        except Exception:
                            old_neu = neutral_prob
                        try:
                            neutral_prob = min(0.95, max(float(neutral_prob), 0.65)) if neutral_prob is not None else 0.65
                        except Exception:
                            neutral_prob = neutral_prob if neutral_prob is not None else 0.65
                        logger.debug(
                            "[REV] reversal cross fired under weak model -> neutral_prob bumped from %s to %.3f",
                            f"{old_neu:.3f}" if isinstance(old_neu, (int, float)) else str(old_neu),
                            float(neutral_prob) if neutral_prob is not None else -1.0,
                        )

                    # Diagnostics for Q
                    model_sign = 1.0 if (buy_prob - 0.5) > 0 else (-1.0 if (buy_prob - 0.5) < 0 else 0.0)
                    ind_sign = np.sign(float(indicator_score)) if indicator_score is not None else 0.0
                    mtf_sign = np.sign(float(mtf.get("mtf_consensus", 0.0))) if mtf else 0.0

                    agree = 0
                    if model_sign != 0 and ind_sign == model_sign:
                        agree += 1
                    if model_sign != 0 and mtf_sign == model_sign:
                        agree += 1

                    # --- Build meta-feature view for Q-model ---
                    try:
                        q_feature_source: Dict[str, float] = {
                            "p_buy": float(buy_prob),
                        }
                        for k in ("cvd_divergence", "vwap_reversion_flag", "wick_extreme_up", "wick_extreme_down"):
                            v = features_raw.get(k, 0.0)
                            try:
                                q_feature_source[k] = float(v)
                            except Exception:
                                q_feature_source[k] = 0.0

                        if q_model_2min is not None:
                            Q_val = q_model_2min.predict_q_hat(buy_prob, q_feature_source)
                            logger.debug(
                                "[Q-MODEL] q_hat=%.3f for p_buy=%.3f | cvd_div=%.3f vwap_rev=%.3f wick_up=%.3f wick_dn=%.3f",
                                Q_val,
                                buy_prob,
                                q_feature_source.get("cvd_divergence", 0.0),
                                q_feature_source.get("vwap_reversion_flag", 0.0),
                                q_feature_source.get("wick_extreme_up", 0.0),
                                q_feature_source.get("wick_extreme_down", 0.0),
                            )
                        else:
                            # Fallback: perfectly neutral quality if no model loaded
                            Q_val = 0.25
                    except Exception as exc:
                        logger.debug("[Q-MODEL] failed to compute q_hat: %s", exc)
                        Q_val = 0.25



                    # --- Exhaustion clamp for trend-continuation trades ---
                    exhausted = False
                    try:
                        z = float(features_raw.get("last_zscore", 0.0))
                        sr_up = float(sr.get("sr_breakout_up", 0.0))
                        sr_dn = float(sr.get("sr_breakout_dn", 0.0))
                        rvol = float(pattern_features.get("pat_rvol", 1.0)) if pattern_features else 1.0

                        # Overextended UP: z > 2, breakout up, high RVOL, and model wants BUY
                        if model_sign > 0 and z > 2.0 and sr_up >= 1.0 and rvol >= 1.5:
                            exhausted = True

                        # Overextended DOWN: z < -2, breakout down, high RVOL, and model wants SELL
                        if model_sign < 0 and z < -2.0 and sr_dn >= 1.0 and rvol >= 1.5:
                            exhausted = True
                    except Exception:
                        exhausted = False

                    if exhausted and model_sign != 0:
                        # Cap Q for continuation in exhausted direction so we never treat it as high edge
                        old_Q = Q_val
                        Q_val = min(Q_val, 0.05)
                        logger.debug(
                            "[%s] [EXHAUST] trend-exhausted (z=%.2f, rvol=%.2f, sr_up=%.1f, sr_dn=%.1f, model_sign=%+.0f) "
                            "→ Q capped %.3f→%.3f",
                            name,
                            float(features_raw.get("last_zscore", 0.0)),
                            float(pattern_features.get("pat_rvol", 1.0)) if pattern_features else 1.0,
                            float(sr.get("sr_breakout_up", 0.0)),
                            float(sr.get("sr_breakout_dn", 0.0)),
                            model_sign,
                            old_Q,
                            Q_val,
                        )
                        
                    # --- End exhaustion clamp ---


                    qmin_base = float(getattr(cfg, "qmin", 0.11)) if hasattr(cfg, "qmin") else 0.11

                    # Neutral-based adjustment (existing behaviour)
                    q_adj = 0.0
                    if neutral_prob is not None:
                        try:
                            neu = float(neutral_prob)
                        except Exception:
                            neu = None
                        if neu is not None:
                            if neu >= 0.70:
                                q_adj += 0.02
                            elif neu <= 0.35:
                                q_adj -= 0.01

                    # Context-based adjustment: trend strength & extension
                    q_ctx = 0.0
                    try:
                        z = float(features_raw.get("last_zscore", 0.0))
                        cons = float(mtf.get("mtf_consensus", 0.0)) if mtf else 0.0

                        # Clean, non-extended trend: slightly easier to trade
                        if abs(z) < 1.0 and abs(cons) >= 0.6:
                            q_ctx -= 0.02

                        # Strong extension (|z| > 2.0): require higher Q
                        if abs(z) > 2.0:
                            q_ctx += 0.02
                    except Exception:
                        q_ctx = 0.0

                    # Confidence-tuner adjustment (only non-zero after enough samples)
                    q_delta = 0.0
                    try:
                        q_delta = float(tuner.qmin_delta())
                    except Exception:
                        q_delta = 0.0

                    qmin_eff = float(np.clip(qmin_base + q_adj + q_ctx + q_delta, 0.06, 0.20))
                    snap = tuner.snapshot()
                    logger.debug(
                        "[CONF] qmin_eff=%.3f (base=%.3f adj=%.3f ctx=%.3f delta=%.3f | hit=%.3f brier=%.3f n=%.0f)",
                        qmin_eff, qmin_base, q_adj, q_ctx, q_delta,
                        snap["hit_rate"], snap["brier"], snap["n"]
                    )


                    margin = abs(buy_prob - 0.5)
                    try:
                        neu_val = float(neutral_prob) if neutral_prob is not None else 0.0
                    except Exception:
                        neu_val = 0.0

                    neutral_gate = 0.60

                    suggest_tradeable = (
                        (margin >= qmin_eff)
                        and (Q_val >= qmin_eff)
                        and (neu_val <= neutral_gate)
                    ) if bool(getattr(cfg, "suggest_tradeable_from_Q", True)) else None

                    tradeable_flag = bool(suggest_tradeable) if (suggest_tradeable is not None) else True

                    # Signal record: prediction for close at t+2*interval
                    sig = {
                        "pred_for": target_start.isoformat(), # where target_start = t + 2*interval
                        "decision": "USER",
                        "buy_prob": float(buy_prob),
                        "sell_prob": float(1.0 - buy_prob),
                        "alpha": None,
                        "neutral_prob": float(neutral_prob) if neutral_prob is not None else None,
                        "mtf_consensus": float(mtf.get("mtf_consensus", 0.0)) if mtf else 0.0,
                        "indicator_score": float(indicator_score) if indicator_score is not None else None,
                        "pattern_adj": float(pattern_features.get('probability_adjustment', 0.0)) if pattern_features else 0.0,
                        "Q": float(Q_val),
                        "qmin_eff": float(qmin_eff),
                        "qmin_delta": float(q_delta),
                        "regime": "na",
                        "suggest_tradeable": suggest_tradeable,
                    }

                    spath = getattr(cfg, "signals_path", "trained_models/production/signals.jsonl")
                    Path(Path(spath).parent).mkdir(parents=True, exist_ok=True)
                    with open(spath, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sig) + "\n")

                    # More informative probability log for the user.
                    # Shows direction, strength (margin), Q, model regime and tradeability.
                    try:
                        model_quality = getattr(model_pipe, "model_quality", "unknown")
                    except Exception:
                        model_quality = "unknown"

                    if buy_prob > 0.5:
                        dir_str = "BUY"
                    elif buy_prob < 0.5:
                        dir_str = "SELL"
                    else:
                        dir_str = "NEUTRAL"

                    margin_val = abs(buy_prob - 0.5)
                    if margin_val >= 0.25:
                        conf_bucket = "HIGH"
                    elif margin_val >= 0.15:
                        conf_bucket = "MED"
                    else:
                        conf_bucket = "LOW"

                    try:
                        neu_val = float(neutral_prob) if neutral_prob is not None else None
                    except Exception:
                        neu_val = None
                    neu_str = f"{neu_val*100:.1f}%" if neu_val is not None else "n/a"

                    logger.info(
                        f"[{name}] Probabilities (2-min): "
                        f"BUY={buy_prob*100:.1f}% | SELL={(1.0-buy_prob)*100:.1f}% | NEU={neu_str} | "
                        f"dir={dir_str} | margin={margin_val:.3f} ({conf_bucket}) | "
                        f"Q={Q_val:.3f} | model={model_quality} | tradeable={tradeable_flag}"
                    )
                    logger.info(
                        f"[{name}] [SIGNAL] 2-min horizon: start={current_bucket_start.strftime('%H:%M:%S')} "
                        f"target={target_start.strftime('%H:%M:%S')} "
                        f"(Q={Q_val:.3f} vs qmin={qmin_eff:.3f}, suggest_tradeable={suggest_tradeable}) → {spath}"
                    )

                    # Stage features for training when t+2 candle closes
                    features_for_log = dict(features)
                    p_raw = getattr(model_pipe, "last_p_xgb_raw", None)
                    p_cal = getattr(model_pipe, "last_p_xgb_calib", None)
                    if p_raw is not None:
                        features_for_log["meta_p_xgb_raw"] = float(p_raw)
                    if p_cal is not None:
                        features_for_log["meta_p_xgb_calib"] = float(p_cal)

                    staged_map[ref_start] = {
                        "features": features_for_log,
                        "buy_prob": float(buy_prob),
                        "alpha": 0.0,
                        "tradeable": tradeable_flag,
                    }

                except Exception as e:
                    logger.error(f"[{name}] on_preclose error: {e}", exc_info=True)

            # ---------- CANDLE CLOSE: 2-MIN LABEL (close_{t+2} vs close_t) ----------
            async def _on_candle_cb(candle_df: pd.DataFrame, full_df: pd.DataFrame):
                try:
                    if not isinstance(candle_df, pd.DataFrame) or candle_df.empty:
                        return

                    idx_ts = candle_df.index[-1]
                    row_t2 = candle_df.iloc[-1]

                    logger.info(
                        f"[{name}] Candle closed at {idx_ts.strftime('%H:%M:%S')} | "
                        f"O:{float(row_t2.get('open', 0.0)):.2f} "
                        f"H:{float(row_t2.get('high', 0.0)):.2f} "
                        f"L:{float(row_t2.get('low', 0.0)):.2f} "
                        f"C:{float(row_t2.get('close', 0.0)):.2f} "
                        f"Vol:{int(row_t2.get('volume', 0)):,} "
                        f"Ticks:{int(row_t2.get('tick_count', 0))}"
                    )

                    # 2-minute horizon: label time t using close at t and close at t+2*interval
                    interval_sec = int(getattr(cfg, 'candle_interval_seconds', 60))
                    interval_min = max(1, interval_sec // 60)
                    horizon_min = 2 * interval_min

                    ref_ts = idx_ts - timedelta(minutes=horizon_min)

                    if not isinstance(full_df, pd.DataFrame) or full_df.empty or ref_ts not in full_df.index:
                        logger.info(
                            f"[{name}] [TRAIN] Skipping 2-min label for {idx_ts.strftime('%H:%M:%S')} "
                            f"(missing reference candle at {ref_ts.strftime('%H:%M:%S')})"
                        )
                        return

                    ref_row = full_df.loc[ref_ts]

                    ref_close = float(ref_row.get('close', 0.0))
                    tgt_close = float(row_t2.get('close', 0.0))
                    if not (np.isfinite(ref_close) and np.isfinite(tgt_close)) or ref_close <= 0.0:
                        logger.info(
                            f"[{name}] [TRAIN] Skipping 2-min label for {idx_ts.strftime('%H:%M:%S')} "
                            f"(invalid closes: ref={ref_close}, tgt={tgt_close})"
                        )
                        return


                    move = tgt_close - ref_close

                    # Simple fixed-percentage tolerance (match offline_train_2min.build_2min_dataset)
                    # tol_pts = flat_tolerance_pct * close_t
                    base_tol_pct = float(getattr(cfg, 'flat_tolerance_pct', 0.00010))
                    tol_pts = float(base_tol_pct * ref_close)

                    is_flat = abs(move) <= tol_pts
                    label = "FLAT" if is_flat else ("BUY" if tgt_close > ref_close else "SELL")


                    # Retrieve staged features for the reference time t
                    staged = staged_map.pop(ref_ts, None)

                    buy_prob = float((staged or {}).get("buy_prob", 0.5))
                    alpha = float((staged or {}).get("alpha", 0.0))
                    tradeable = bool((staged or {}).get("tradeable", True))
                    features_for_log = dict((staged or {}).get("features", {}))

                    # Update rolling confidence tuner for directional labels
                    try:
                        dir_true = 1 if label == "BUY" else (-1 if label == "SELL" else (0 if label == "FLAT" else None))
                        if dir_true is not None:
                            tuner.update(dir_true=dir_true, buy_prob=buy_prob)
                    except Exception as e:
                        logger.debug(f"[CONF] tuner update failed (ignored): {e}")

                    # Write training row timestamped at ref_ts (start of 2-min horizon)

                    cols = [
                        idx_ts.isoformat(),  # use target close time so it matches signals.pred_for
                        "USER",
                        label,
                        f"{buy_prob:.6f}",
                        f"{alpha:.6f}",
                        str(tradeable),
                        str(is_flat),
                        str(int(row_t2.get('tick_count', 0)))
                    ]

                    
                    for k in sorted(features_for_log.keys()):
                        try:
                            cols.append(f"{k}={float(features_for_log[k]):.8f}")
                        except Exception:
                            continue

                    with open(feature_log_path, "a", encoding="utf-8") as f:
                        f.write(",".join(cols) + "\n")

                    logger.info(
                        f"[{name}] [TRAIN] Logged 2-min features for ref={ref_ts.strftime('%H:%M:%S')} "
                        f"-> tgt={idx_ts.strftime('%H:%M:%S')} label={label} tradeable={tradeable}"
                    )

                except Exception as e:
                    logger.error(f"[{name}] on_candle callback error: {e}", exc_info=True)

            # Wire callbacks
            ws_handler.on_tick = _on_tick_cb
            ws_handler.on_preclose = _on_preclose_cb
            ws_handler.on_candle = _on_candle_cb

            # Start connector and heartbeat
            stop_event = asyncio.Event()
            stop_events[name] = stop_event
            ws_task = asyncio.create_task(_ws_connect_and_stream(name, cfg, ws_handler, stop_event))
            hb_task = asyncio.create_task(_websocket_heartbeat(name, ws_handler, interval_sec=30))
            ws_tasks.append((name, ws_handler, ws_task))
            hb_tasks.append((name, hb_task))
            logger.info(f"[{name}] Connector and heartbeat started")

        except Exception as e:
            logger.error(f"[{name}] Failed to initialize connection: {e}", exc_info=True)

    if not ws_tasks:
        logger.critical("No connections could be initialized. Exiting.")
        try:
            if trainer_task:
                trainer_task.cancel()
                await asyncio.wait_for(trainer_task, timeout=5.0)
        except Exception:
            pass
        return

    # Keep main alive until session guard requests stop
    stop_main_event = asyncio.Event()

    async def _session_guard():
        try:
            def _today_ist(h, m):
                now = datetime.now(IST)
                return now.replace(hour=h, minute=m, second=0, microsecond=0)

            roll_ts = _today_ist(15, 10)
            exit_ts = _today_ist(15, 15)

            now = datetime.now(IST)
            if now > exit_ts:
                roll_ts = roll_ts + timedelta(days=1)
                exit_ts = exit_ts + timedelta(days=1)

            await asyncio.sleep(max(0.0, (roll_ts - datetime.now(IST)).total_seconds()))
            try:
                hist_path = os.getenv("FEATURE_LOG_HIST", "trained_models/production/feature_log_hist.csv")
                
                _roll_feature_logs(
                    daily_path=getattr(config, 'feature_log_path', 'feature_log.csv'),
                    hist_path=hist_path,
                    cap_rows=2000
                )
                logger.info("[EOD] Rolled daily feature log into historical and capped to 2000 rows")
            except Exception as e:
                logger.warning(f"[EOD] Roll failed: {e}")

            await asyncio.sleep(max(0.0, (exit_ts - datetime.now(IST)).total_seconds()))
            logger.info("[EOD] Session end reached (15:15 IST). Requesting shutdown.")
            stop_main_event.set()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[EOD] Session guard error: {e}")

    def _roll_feature_logs(daily_path: str, hist_path: str, cap_rows: int = 2000) -> None:
        try:
            daily_lines = []
            if os.path.exists(daily_path):
                with open(daily_path, "r", encoding="utf-8") as f:
                    daily_lines = [ln.strip() for ln in f if ln.strip()]
            hist_lines = []
            if os.path.exists(hist_path):
                with open(hist_path, "r", encoding="utf-8") as f:
                    hist_lines = [ln.strip() for ln in f if ln.strip()]

            seen = set()
            merged = []
            for ln in hist_lines + daily_lines:
                ts = ln.split(",", 1)[0].strip()
                if ts and ts not in seen:
                    seen.add(ts)
                    merged.append(ln)

            if cap_rows > 0 and len(merged) > cap_rows:
                merged = merged[-cap_rows:]

            with open(hist_path, "w", encoding="utf-8") as f:
                f.write("\n".join(merged) + "\n")
        except Exception as e:
            raise e

    session_task = asyncio.create_task(_session_guard())

    try:
        while not stop_main_event.is_set():
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Main event loop cancelled")
    except Exception as e:
        logger.error(f"Fatal error in main_loop: {e}", exc_info=True)
    finally:
        logger.info("Shutting down main loop")

        for name, _handler, _task in ws_tasks:
            try:
                stop_events[name].set()
            except Exception:
                pass

        for name, _handler, ws_task in ws_tasks:
            ws_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(ws_task, timeout=5.0)

        for name, hb_task in hb_tasks:
            hb_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(hb_task, timeout=3.0)

        if trainer_task:
            trainer_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(trainer_task, timeout=5.0)

        if baseline_task:
            baseline_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(baseline_task, timeout=5.0)

        if calib_task:
            calib_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(calib_task, timeout=5.0)

        if session_task:
            session_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(session_task, timeout=5.0)

        logger.info("=" * 60)
        logger.info("MAIN EVENT LOOP SHUTDOWN COMPLETE")
        logger.info("=" * 60)
