# main_event_loop.py
import asyncio
import base64
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
            # Basic iterable support
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
    # Align schema to the booster right away (prevents schema drift)
    try:
        model_pipe.replace_models(xgb=xgb)
        logger.info("Pipeline schema aligned to booster at startup (if embedded)")
    except Exception as e:
        logger.warning(f"Pipeline schema alignment skipped at startup: {e}")
    logger.info("Global components initialized successfully")

    # Paths
    feature_log_path = getattr(config, 'feature_log_path', 'feature_log.csv')
    logger.info(f"Feature log path: {feature_log_path}")

    # Background online trainer (kept)
    trainer_task = None
    try:
        from online_trainer import background_trainer_loop
        xgb_out = os.getenv("XGB_PATH", "models/xgb_model.json")
        neutral_out = os.getenv("NEUTRAL_PATH", "models/neutral_model.pkl")
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
        calib_out = os.getenv("CALIB_PATH", "models/platt_calibration.json")
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

            # Per-connection state
            ob_ring = deque(maxlen=10)                        # order book snapshots (for spread calc)
            staged_map: Dict[datetime, Dict[str, Any]] = {}   # features staged per next candle start

            # on_tick: capture a lightweight OB snapshot for spread calc and any UI needs
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

            # on_preclose: predict ONCE per bucket (handled by ws_handler guard)
            async def _on_preclose_cb(preview_df: pd.DataFrame, full_df: pd.DataFrame):
                try:
                    # Determine current bucket start and next candle start
                    if preview_df is None or preview_df.empty:
                        return
                    current_bucket_start = preview_df.index[-1]
                    interval_sec = int(getattr(cfg, 'candle_interval_seconds', 60))
                    interval_min = max(1, interval_sec // 60)
                    next_candle_start = current_bucket_start + timedelta(minutes=interval_min)

                    # Gather prices for TA/indicator scaling
                    try:
                        prices = ws_handler.get_prices(last_n=200) if hasattr(ws_handler, 'get_prices') else []
                    except Exception:
                        prices = []
                    if not prices or len(prices) < 2:
                        return

                    # Compute TA/EMA
                    ema_feats = FeaturePipeline.compute_emas(prices)
                    ta = {}
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

                    # Micro features
                    try:
                        micro_ctx = ws_handler.get_micro_features()
                        px_arr = np.asarray(prices[-64:], dtype=float)
                        last_px = float(px_arr[-1]) if px_arr.size else 0.0
                        # Normalize micro_slope by realized short vol to keep it scale-safe
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

                    # Indicator score (context only)
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

                    # Feature map
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

                    features_raw = {
                        **ema_feats,
                        **ofd,
                        'micro_slope': micro_slope_normed,
                        'micro_imbalance': float(micro_ctx.get('imbalance', 0.0)),
                        'mean_drift_pct': float(micro_ctx.get('mean_drift_pct', 0.0)),
                        'last_price': float(prices[-1]) if prices else 0.0,
                        'last_zscore': float(last_zscore),
                        'std_dltp_short': float(micro_ctx.get('std_dltp_short', 0.0)),
                        'price_range_tightness': float(micro_ctx.get('price_range_tightness', 0.0)),
                        **ta,
                        **sr,
                    }
                    features = FeaturePipeline.normalize_features(features_raw, scale=scale)
                    for k, v in (pattern_features or {}).items():
                        features[k] = float(v)
                    for k, v in (mtf or {}).items():
                        features[k] = float(v)

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

                    # Predict (probabilities only)
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

                    # Diagnostics Q and qmin (for optional UI suggestion)
                    model_sign = 1.0 if (buy_prob - 0.5) > 0 else (-1.0 if (buy_prob - 0.5) < 0 else 0.0)
                    ind_sign = np.sign(float(indicator_score)) if indicator_score is not None else 0.0
                    mtf_sign = np.sign(float(mtf.get("mtf_consensus", 0.0))) if mtf else 0.0
                    agree = 0
                    if model_sign != 0 and ind_sign == model_sign:
                        agree += 1
                    if model_sign != 0 and mtf_sign == model_sign:
                        agree += 1
                    A = (1 + agree) / 3.0
                    Q_val = abs(buy_prob - 0.5) * A
                    qmin_base = float(getattr(cfg, "qmin", 0.11)) if hasattr(cfg, "qmin") else 0.11
                    q_adj = 0.0
                    if neutral_prob is not None:
                        if neutral_prob >= 0.70:
                            q_adj += 0.02
                        elif neutral_prob <= 0.35:
                            q_adj -= 0.01
                    qmin_eff = float(np.clip(qmin_base + q_adj, 0.06, 0.20))

                    # Emit signal for next candle (decision=USER)
                    suggest_tradeable = bool(Q_val >= qmin_eff) if bool(getattr(cfg, "suggest_tradeable_from_Q", True)) else None
                    sig = {
                        "pred_for": next_candle_start.isoformat(),
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
                        "regime": "na",
                        "suggest_tradeable": suggest_tradeable
                    }
                    spath = getattr(cfg, "signals_path", "logs/signals.jsonl")
                    Path(Path(spath).parent).mkdir(parents=True, exist_ok=True)
                    with open(spath, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sig) + "\n")
                    logger.info(f"[{name}] Probabilities: BUY={buy_prob*100:.1f}% | SELL={(1.0-buy_prob)*100:.1f}% (no auto-decision)")
                    logger.info(f"[{name}] [SIGNAL] Wrote signal for {next_candle_start.strftime('%H:%M:%S')} "
                                f"(Q={Q_val:.3f} vs qmin={qmin_eff:.3f}, suggest_tradeable={suggest_tradeable}) → {spath}")

                    # Stage features/training info for candle-close logger
                    features_for_log = dict(features)
                    p_raw = getattr(model_pipe, "last_p_xgb_raw", None)
                    p_cal = getattr(model_pipe, "last_p_xgb_calib", None)
                    if p_raw is not None:
                        features_for_log["meta_p_xgb_raw"] = float(p_raw)
                    if p_cal is not None:
                        features_for_log["meta_p_xgb_calib"] = float(p_cal)
                    staged_map[next_candle_start] = {
                        "features": features_for_log,
                        "buy_prob": float(buy_prob),
                        "alpha": 0.0,  # reference only
                        "tradeable": True
                    }
                except Exception as e:
                    logger.error(f"[{name}] on_preclose error: {e}", exc_info=True)

            # on_candle: label last candle and log one row for training
            async def _on_candle_cb(candle_df: pd.DataFrame, full_df: pd.DataFrame):
                try:
                    idx_ts = candle_df.index[-1] if (isinstance(candle_df, pd.DataFrame) and not candle_df.empty) else None
                    if idx_ts is None:
                        return
                    row = candle_df.iloc[-1]
                    logger.info(
                        f"[{name}] Candle closed at {idx_ts.strftime('%H:%M:%S')} | "
                        f"O:{float(row.get('open', 0.0)):.2f} "
                        f"H:{float(row.get('high', 0.0)):.2f} "
                        f"L:{float(row.get('low', 0.0)):.2f} "
                        f"C:{float(row.get('close', 0.0)):.2f} "
                        f"Vol:{int(row.get('volume', 0)):,} "
                        f"Ticks:{int(row.get('tick_count', 0))}"
                    )
                    # Compute label with dynamic flat tolerance
                    o = float(row.get('open', 0.0)); c = float(row.get('close', 0.0))
                    move = c - o
                    base_tol = float(getattr(cfg, 'flat_tolerance_pct', 0.0002))
                    price_denom = max(1e-6, o)
                    rng_points = float(row.get('high', o)) - float(row.get('low', o))
                    rng_pct = rng_points / price_denom
                    try:
                        closes = full_df['close'].astype(float).tail(20).values if hasattr(full_df, 'columns') else []
                        vol_pct = (np.std(np.diff(closes)) / price_denom) if len(closes) >= 3 else 0.0
                    except Exception:
                        vol_pct = 0.0
                    k_range = float(getattr(cfg, 'flat_dyn_k_range', 0.20))
                    min_pts = float(getattr(cfg, 'flat_min_points', 0.20))
                    dyn_raw = (k_range * rng_pct) + (0.50 * vol_pct)
                    dyn_tol = max(0.5 * base_tol, min(base_tol, dyn_raw))
                    tol_pts = max(min_pts, dyn_tol * price_denom)
                    tol_cap_pts = float(getattr(cfg, 'flat_tolerance_max_pts', 6.00))
                    tol_pts = min(tol_pts, tol_cap_pts) if tol_cap_pts > 0 else tol_pts
                    is_flat = abs(move) <= tol_pts
                    label = "FLAT" if is_flat else ("BUY" if c > o else "SELL")

                    staged = staged_map.pop(idx_ts, None)
                    buy_prob = float((staged or {}).get("buy_prob", 0.5))
                    alpha = float((staged or {}).get("alpha", 0.0))
                    tradeable = bool((staged or {}).get("tradeable", True))
                    features_for_log = dict((staged or {}).get("features", {}))

                    cols = [
                        idx_ts.isoformat(), "USER", label, f"{buy_prob:.6f}", f"{alpha:.6f}",
                        str(tradeable), str(is_flat), str(int(row.get('tick_count', 0)))
                    ]
                    for k in sorted(features_for_log.keys()):
                        try:
                            cols.append(f"{k}={float(features_for_log[k]):.8f}")
                        except Exception:
                            continue
                    with open(feature_log_path, "a", encoding="utf-8") as f:
                        f.write(",".join(cols) + "\n")
                    logger.info(f"[{name}] [TRAIN] Logged features for {idx_ts.strftime('%H:%M:%S')} label={label} tradeable={tradeable}")
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
            # Compute today's cutoff times in IST
            def _today_ist(h, m):
                now = datetime.now(IST)
                return now.replace(hour=h, minute=m, second=0, microsecond=0)

            roll_ts = _today_ist(15, 10)
            exit_ts = _today_ist(15, 15)

            # If running past cutoff (testing), schedule for next day
            now = datetime.now(IST)
            if now > exit_ts:
                roll_ts = roll_ts + timedelta(days=1)
                exit_ts = exit_ts + timedelta(days=1)

            # Sleep until 15:10, then roll logs
            await asyncio.sleep(max(0.0, (roll_ts - datetime.now(IST)).total_seconds()))
            try:
                _roll_feature_logs(
                    daily_path=getattr(config, 'feature_log_path', 'feature_log.csv'),
                    hist_path="feature_log_hist.csv",
                    cap_rows=2000
                )
                logger.info("[EOD] Rolled daily feature log into historical and capped to 2000 rows")
            except Exception as e:
                logger.warning(f"[EOD] Roll failed: {e}")

            # Sleep until 15:15, then request stop
            await asyncio.sleep(max(0.0, (exit_ts - datetime.now(IST)).total_seconds()))
            logger.info("[EOD] Session end reached (15:15 IST). Requesting shutdown.")
            stop_main_event.set()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[EOD] Session guard error: {e}")

    # Helper to roll logs with de-dup and capping
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

            # Merge and de-duplicate by timestamp (first CSV column)
            seen = set()
            merged = []
            for ln in hist_lines + daily_lines:
                ts = ln.split(",", 1)[0].strip()
                if ts and ts not in seen:
                    seen.add(ts)
                    merged.append(ln)

            # Keep only the last cap_rows rows
            if cap_rows > 0 and len(merged) > cap_rows:
                merged = merged[-cap_rows:]

            with open(hist_path, "w", encoding="utf-8") as f:
                f.write("\n".join(merged) + "\n")
        except Exception as e:
            raise e

    # Start session guard
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
        # Stop WS connections
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



        # Stop session guard
        if session_task:
            session_task.cancel()
            with suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(session_task, timeout=5.0)



        logger.info("=" * 60)
        logger.info("MAIN EVENT LOOP SHUTDOWN COMPLETE")
        logger.info("=" * 60)
