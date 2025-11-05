# main_event_loop.py

"""
Concurrent main event loop for AR-NMS (volume-free, ATR-free).

Key points:
- NO import of enhanced_websocket_handler.py (reference-only).
- Uses core_handler.UnifiedWebSocketHandler for state and parsing.
- Implements DhanHQ v2 connection / subscription / message processing here.
- Supports multiple concurrent connections (config.connections).
- Async callbacks push ticks into per-connection queues for processing.
- Heartbeats and data-stall watchdog for liveness and auto-recovery.
- Global weight refresh task and graceful shutdown.

To get live ticks:
- Ensure DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID env vars are set as plaintext.
- run_main.py base64-encodes them into config (dhan_access_token_b64, dhan_client_id_b64).
- This loop builds the URL: wss://api-feed.dhan.co?version=2&token={access_token}&clientId={client_id}&authType=2
"""

import asyncio
import base64
import json
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import numpy as np
import websockets
import pandas as pd

from core_handler import UnifiedWebSocketHandler as WSHandler
from feature_pipeline import FeaturePipeline
from model_pipeline import create_default_pipeline, weight_refresh_loop
from telegram_bot import TelegramBot

logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


# ------------------------- Helper ------------------------- #

def _install_preclose_shim_if_missing(ws_handler, cfg):
    """Install pre-close callback shim if handler doesn't have it."""
    if hasattr(ws_handler, "_maybe_fire_preclose"):
        return  # Already installed
    
    async def _maybe_fire_preclose(self, now_ts):
        try:
            # Ensure callback and active candle
            if not getattr(self, "on_preclose", None):
                return
            start = self.current_candle.get("start_time") if isinstance(self.current_candle, dict) else None
            if start is None:
                return
            
            # Compute pre-close window
            lead = int(getattr(self.config, "preclose_lead_seconds", 10))
            interval_min = max(1, int(getattr(self.config, "candle_interval_seconds", 60)) // 60)
            close_time = start + timedelta(minutes=interval_min)
            preclose_time = close_time - timedelta(seconds=lead)
            
            # Conditions: in lead window, not fired yet, and have ticks
            if now_ts < preclose_time:
                return
            if getattr(self, "_preclose_fired_for_bucket", None) == start:
                return
            
            ticks = self.current_candle.get("ticks") or []
            prices = [float(t.get("ltp", 0.0)) for t in ticks if float(t.get("ltp", 0.0)) > 0.0]
            if not prices:
                return
            
            # Assemble a minimal OHLC preview DataFrame
            o, h, l, c = prices[0], max(prices), min(prices), prices[-1]
            preview = pd.DataFrame([{
                "timestamp": start,
                "open": o, "high": h, "low": l, "close": c,
                "volume": 0, "tick_count": len(prices)
            }]).set_index("timestamp")
            
            # Mark fired before awaiting callback to avoid double-fire races
            self._preclose_fired_for_bucket = start
            
            # Hand off to your on_preclose (second arg is entire candle history if available)
            hist = getattr(self, "candle_data", None)
            await self.on_preclose(preview, hist.copy() if hasattr(hist, "copy") and hist is not None else preview.copy())
        except Exception:
            # Be silent here to avoid noisy logs during lead window
            pass
    
    # Bind the async method to the instance
    ws_handler._maybe_fire_preclose = _maybe_fire_preclose.__get__(ws_handler, ws_handler.__class__)
    logger.info("Installed pre-close shim on UnifiedWebSocketHandler")


# ------------------------- Utilities ------------------------- #

async def _websocket_heartbeat(name: str, ws_handler: WSHandler, interval_sec: int = 30):
    """Periodic health log per connection."""
    while True:
        try:
            ticks = getattr(ws_handler, "tick_count", 0)
            last_ts = getattr(ws_handler, "last_packet_time", None)
            if last_ts and hasattr(last_ts, "timestamp"):
                age = max(0.0, time.time() - last_ts.timestamp())
                logger.info(f"[{name}] Websocket is active. ticks={ticks}, last_packet_age={age:.1f}s")
            else:
                logger.info(f"[{name}] Websocket is active. ticks={ticks}, awaiting first packet...")
        except asyncio.CancelledError:
            logger.info(f"[{name}] Heartbeat cancelled")
            break
        except Exception as e:
            logger.debug(f"[{name}] Heartbeat error (ignored): {e}")
        finally:
            await asyncio.sleep(interval_sec)


def _build_live_tensor_from_prices(prices: List[float], length: int = 64) -> np.ndarray:
    """Build normalized price tensor from history (fallback if handler tensor missing)."""
    try:
        px = np.array(prices[-length:], dtype=float)
        if px.size == 0:
            return np.zeros((1, length, 1), dtype=float)
        px = (px - px.mean()) / max(1e-9, px.std())
        if px.size < length:
            pad = np.zeros(length - px.size, dtype=float)
            px = np.concatenate([pad, px])
        else:
            px = px[-length:]
        return px.reshape(1, length, 1)
    except Exception:
        return np.zeros((1, length, 1), dtype=float)


def _extract_best_and_mid_from_tick(tick: Dict[str, Any]) -> Tuple[float, float]:
    """Extract best executable price and mid price; fallback to LTP."""
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


def _normalize_connections(config) -> List[Tuple[str, Any]]:
    """
    Normalize config into a list of (name, cfg):
    - If config.connections is dict: use its names.
    - If list: synthesize names.
    - Else: single 'primary' connection from config.
    """
    conns: List[Tuple[str, Any]] = []
    try:
        if hasattr(config, "connections") and isinstance(config.connections, dict):
            for name, cfg in config.connections.items():
                conns.append((str(name), cfg))
        elif hasattr(config, "connections") and isinstance(config.connections, Iterable):
            for i, cfg in enumerate(config.connections, start=1):
                sec = getattr(cfg, "nifty_security_id", getattr(cfg, "security_id", "NA"))
                seg = getattr(cfg, "nifty_exchange_segment", getattr(cfg, "exchange_segment", "NA"))
                conns.append((f"conn{i}:{seg}:{sec}", cfg))
        else:
            sec = getattr(config, "nifty_security_id", getattr(config, "security_id", "NA"))
            seg = getattr(config, "nifty_exchange_segment", getattr(config, "exchange_segment", "NA"))
            conns.append((f"primary:{seg}:{sec}", config))
    except Exception:
        conns.append(("primary", config))
    return conns if conns else [("primary", config)]


def _build_dhan_ws_url(cfg: Any) -> Optional[str]:
    """
    Build DhanHQ v2 WS URL using base64-encoded credentials from cfg.
    Returns None if creds missing.
    """
    try:
        tok_b64 = getattr(cfg, "dhan_access_token_b64", "") or ""
        cid_b64 = getattr(cfg, "dhan_client_id_b64", "") or ""
        if not tok_b64 or not cid_b64:
            return None
        access_token = base64.b64decode(tok_b64).decode("utf-8")
        client_id = base64.b64decode(cid_b64).decode("utf-8")
        return (
            "wss://api-feed.dhan.co"
            f"?version=2&token={access_token}&clientId={client_id}&authType=2"
        )
    except Exception as e:
        logger.error(f"Failed to build Dhan WS URL: {e}")
        return None


def _subscription_payload(cfg: Any) -> Dict[str, Any]:
    """
    Create DhanHQ v2 subscription payload for a single instrument.
    """
    return {
        "RequestCode": 15,
        "InstrumentCount": 1,
        "InstrumentList": [{
            "ExchangeSegment": getattr(cfg, "nifty_exchange_segment", "IDX_I"),
            "SecurityId": str(getattr(cfg, "nifty_security_id", "")),
        }]
    }


# ------------------------- WebSocket Connectors ------------------------- #

async def _data_stall_watchdog(
    name: str,
    ws_handler: WSHandler,
    resubscribe_cb,
    reconnect_cb,
    stall_secs: int,
    reconnect_secs: int
):
    """
    Monitor for data stalls. If no packets arrive after subscribe:
    - After stall_secs: resubscribe once.
    - After reconnect_secs: force reconnect.
    """
    did_resubscribe = False
    last_sub_time: Optional[datetime] = None

    # Provide a way for outer code to update last_sub_time
    def set_last_sub_time(ts: datetime):
        nonlocal last_sub_time
        last_sub_time = ts

    # Expose setter to outer closures
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
                # Received data: reset resubscribe flag
                did_resubscribe = False
        except asyncio.CancelledError:
            logger.debug(f"[{name}] Watchdog cancelled")
            break
        except Exception as e:
            logger.error(f"[{name}] Watchdog error: {e}", exc_info=True)
            await asyncio.sleep(2)


async def _ws_connect_and_stream(
    name: str,
    cfg: Any,
    ws_handler: WSHandler,
    stop_event: asyncio.Event
):
    """
    Connect to DhanHQ v2 WS, subscribe, and stream messages.
    Uses ws_handler's parsing for 16-byte TICKER packets and pushes ticks into handler.
    Auto-reconnects with exponential backoff.
    """
    backoff_base = int(getattr(cfg, "reconnect_delay_base", 2)) or 2
    max_attempts = int(getattr(cfg, "max_reconnect_attempts", 5)) or 5

    while not stop_event.is_set():
        ws_url = _build_dhan_ws_url(cfg)
        if not ws_url:
            logger.critical(f"[{name}] Missing or invalid credentials; cannot build WS URL")
            return

        attempt = 0
        while attempt < max_attempts and not stop_event.is_set():
            attempt += 1
            backoff = min(backoff_base * (2 ** (attempt - 1)), 60)
            try:
                logger.info(f"[{name}] Connecting to {ws_url} (attempt {attempt}/{max_attempts})")
                async with websockets.connect(
                    ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,
                    compression=None
                ) as ws:
                    logger.info(f"[{name}] WebSocket connected")

                    # Subscription
                    sub = _subscription_payload(cfg)
                    last_sub_time = datetime.now(IST)
                    await ws.send(json.dumps(sub))
                    logger.info(f"[{name}] Subscription sent at {last_sub_time.strftime('%H:%M:%S')} | "
                                f"RequestCode={sub['RequestCode']} "
                                f"ExchangeSegment={sub['InstrumentList'][0]['ExchangeSegment']} "
                                f"SecurityId={sub['InstrumentList'][0]['SecurityId']}")

                    # Watchdog to resubscribe/reconnect on stall
                    stall_secs = int(getattr(cfg, "data_stall_seconds", 15)) or 15
                    reconn_secs = int(getattr(cfg, "data_stall_reconnect_seconds", 30)) or 30

                    async def resubscribe():
                        nonlocal last_sub_time
                        await ws.send(json.dumps(sub))
                        last_sub_time = datetime.now(IST)
                        logger.info(f"[{name}] Resubscription sent at {last_sub_time.strftime('%H:%M:%S')}")

                    async def reconnect():
                        # Force close; context manager will exit
                        try:
                            await ws.close()
                        except Exception:
                            pass

                    watchdog_task = asyncio.create_task(
                        _data_stall_watchdog(
                            name=name,
                            ws_handler=ws_handler,
                            resubscribe_cb=resubscribe,
                            reconnect_cb=reconnect,
                            stall_secs=stall_secs,
                            reconnect_secs=reconn_secs
                        )
                    )

                    # Allow watchdog to be aware of last_subscribe_time
                    setter = getattr(reconnect, "_set_last_sub_time", None)
                    if callable(setter):
                        setter(last_sub_time)

                    # Message loop
                    message_count = 0
                    try:
                        async for message in ws:
                            if stop_event.is_set():
                                break
                            message_count += 1

                            if isinstance(message, bytes):
                                # Use handler's ticker parser when applicable
                                tick_data = None
                                try:
                                    if len(message) == 16 and message and message[0] == ws_handler.TICKER_PACKET:
                                        if hasattr(ws_handler, "_parse_ticker_packet"):
                                            tick_data = ws_handler._parse_ticker_packet(message)
                                    # Optional: support other packet sizes if you add parsers in core_handler
                                    # elif len(message) >= 50 and message[0] == ws_handler.QUOTE_PACKET and hasattr(ws_handler, "_parse_quote_packet"):
                                    #     tick_data = ws_handler._parse_quote_packet(message)
                                    # elif len(message) == 162 and message[0] == ws_handler.FULL_PACKET and hasattr(ws_handler, "_parse_full_packet"):
                                    #     tick_data = ws_handler._parse_full_packet(message)
                                except Exception as e:
                                    logger.error(f"[{name}] Parse error: {e}", exc_info=True)

                                if tick_data:
                                    await ws_handler._process_tick(tick_data)

                            else:
                                # Control text frame - best-effort log
                                try:
                                    data = json.loads(message)
                                    code = data.get("ResponseCode")
                                    if code:
                                        logger.info(f"[{name}] Control: code={code} msg={data.get('ResponseMessage', '')}")
                                except Exception:
                                    logger.debug(f"[{name}] Text message: {str(message)[:200]}")

                            # Update watchdog subscribe time if needed (no ticks received yet)
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

                # If we exit the context without stop_event, reconnect
                if stop_event.is_set():
                    return
                logger.warning(f"[{name}] Message loop ended; reconnecting after {backoff}s")
                await asyncio.sleep(backoff)

            except asyncio.CancelledError:
                logger.info(f"[{name}] Connector cancelled")
                return
            except Exception as e:
                logger.error(f"[{name}] Connect attempt failed: {e}")
                if attempt < max_attempts:
                    logger.info(f"[{name}] Retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                else:
                    logger.critical(f"[{name}] Failed to establish WS after {max_attempts} attempts")
                    return

        # If attempts exhausted
        if stop_event.is_set():
            return
        # Pause before a new round of attempts
        await asyncio.sleep(1)


# ------------------------- Per-Connection Processing ------------------------- #
async def _connection_processing_loop(
    name: str,
    ws_handler: WSHandler,
    feat_pipe: FeaturePipeline,
    model_pipe,
    telegram: TelegramBot,
    config: Any,
    tick_queue: "asyncio.Queue[Dict[str, Any]]",
    ob_ring: deque,
    pending_preds: Dict[datetime, Dict[str, float]],
    feature_log_path: str,
    pending_train_rows: Dict[datetime, Dict[str, Any]]  
):

    """
    Consume ticks from tick_queue for this connection, compute features,
    run predictions, and make execution decisions. Emits periodic summaries.
    
    NEW FEATURES:
    - No-trade gate for micro-flat/illiquid minutes
    - Fair HOLD/FLAT scoring with flat_tolerance_pct
    - Rolling PF tracking with record_pnl()
    - Feature + latent + label logging to feature_log.csv
    """

    tick_counter = 0
    interval_min = max(1, int(getattr(config, 'candle_interval_seconds', 60)) // 60)
    current_bucket_start: Optional[datetime] = None
    last_minute_prediction = None

    # Pending training rows and staged row for current bucket
    # pending_train_rows: Dict[datetime, Dict[str, Any]] = {}
    staged_row_current_bucket: Optional[Dict[str, Any]] = None

    def _make_decision(buy_prob: float, alpha: float) -> str:
        sell_prob = 1.0 - float(buy_prob)
        if buy_prob >= alpha:
            return "BUY"
        if sell_prob >= alpha:
            return "SELL"
        return "HOLD"

    logger.info(f"[{name}] Processing loop started")

    try:
        while True:
            tick = await tick_queue.get()
            tick_counter += 1

            # # Uncomment during deep debug to surface every 50th tick
            # if tick_counter == 1 or (tick_counter % 50 == 0):
            #     ltp_val = float(tick.get('ltp', 0.0))
            #     ts = tick.get('timestamp')
            #     ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
            #     logger.info(f"[{name}] Handler received tick #{tick_counter}: ltp={ltp_val:.4f}, ts={ts_str}")


            timestamp = tick.get('timestamp')
            if not isinstance(timestamp, datetime):
                timestamp = datetime.now(IST)
            bucket_minute = (timestamp.minute // interval_min) * interval_min
            tick_bucket_start = timestamp.replace(minute=bucket_minute, second=0, microsecond=0)

            if current_bucket_start is None:
                current_bucket_start = tick_bucket_start
            elif tick_bucket_start != current_bucket_start:
                try:
                    if last_minute_prediction is not None:
                        next_candle_start = current_bucket_start + timedelta(minutes=interval_min)
                        logger.info(
                            f"[{name}] Predicted next {interval_min}m candle: start={next_candle_start.isoformat()}, "
                            f"decision={last_minute_prediction['decision']}, "
                            f"buy_prob={last_minute_prediction['buy_prob']:.3f}, "
                            f"sell_prob={1.0 - last_minute_prediction['buy_prob']:.3f}, "
                            f"alpha={last_minute_prediction['alpha']:.3f}, "
                            f"last_ltp={float(last_minute_prediction['ltp']):.4f}"
                        )
                        
                        # Percentage-format probabilities and threshold
                        bp = float(last_minute_prediction["buy_prob"])
                        bp = 0.0 if not np.isfinite(bp) else min(max(bp, 0.0), 1.0)
                        sp = 1.0 - bp
                        alpha_v = float(last_minute_prediction["alpha"])
                        alpha_v = 0.0 if not np.isfinite(alpha_v) else min(max(alpha_v, 0.0), 1.0)
            
                        logger.info(
                            f"[{name}] Decision probabilities: BUY={bp*100:.1f}% | SELL={sp*100:.1f}% | threshold={alpha_v*100:.1f}% → {last_minute_prediction['decision']}"
                        )

                        # Register prediction for next candle
                        try:
                            if next_candle_start not in pending_preds:
                                pending_preds[next_candle_start] = dict(last_minute_prediction)
                            else:
                                pending_preds[next_candle_start].update(last_minute_prediction)
                            logger.info(
                                f"[{name}] [EVAL] Registered prediction for {next_candle_start.strftime('%H:%M:%S')} → "
                                f"{last_minute_prediction['decision']} (bp={last_minute_prediction['buy_prob']:.3f}, "
                                f"thr={last_minute_prediction['alpha']:.3f})"
                            )
                        except Exception as e:
                            logger.debug(f"[{name}] [EVAL] Failed to register prediction at rollover: {e}")

                        # Stage training row for the next candle (label will be known at close)
                        try:
                            if staged_row_current_bucket is not None:
                                pending_train_rows[next_candle_start] = dict(staged_row_current_bucket)
                                logger.info(
                                    f"[{name}] [TRAIN] Staged features for {next_candle_start.strftime('%H:%M:%S')} "
                                    f"(decision={staged_row_current_bucket['decision']} bp={staged_row_current_bucket['buy_prob']:.3f} thr={staged_row_current_bucket['alpha']:.3f} "
                                    f"no_trade={staged_row_current_bucket['no_trade']})"
                                )
                                # PF visibility at rollover
                                try:
                                    pf_now = ws_handler.get_recent_profit_factor()
                                    logger.info(f"[{name}] [PF] Rolling PF={float(pf_now):.3f} → next alpha may adapt")
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.debug(f"[{name}] [TRAIN] Failed to stage training row: {e}")
                                    
                except Exception:
                    pass
                current_bucket_start = tick_bucket_start
                last_minute_prediction = None

            # Visibility
            try:
                if tick_counter == 1 or (tick_counter % 10 == 0):
                    ltp_val = float(tick.get('ltp', 0.0))
                    ts_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
            except Exception:
                pass

            # Prices
            try:
                prices = ws_handler.get_prices(last_n=200) if hasattr(ws_handler, 'get_prices') else []
            except Exception:
                prices = []
            if not prices or len(prices) < 2:
                await asyncio.sleep(0)
                continue

            # Features
            ema_feats = FeaturePipeline.compute_emas(prices)
            
            
            try:
                order_books = []
                if hasattr(ws_handler, 'get_order_books'):
                    order_books = ws_handler.get_order_books(last_n=5) or []
                if not order_books:
                    order_books = list(ob_ring)
            except Exception:
                order_books = list(ob_ring)

                
                
            ofd = FeaturePipeline.order_flow_dynamics(order_books)

            try:
                micro_ctx = ws_handler.get_micro_features()
            except Exception:
                micro_ctx = {}



            # Initialize defaults to avoid UnboundLocalError
            n_short = int(micro_ctx.get('n_short', 0))
            std_short = float(micro_ctx.get('std_dltp_short', 0.0))
            tightness = float(micro_ctx.get('price_range_tightness', 0.0))
            gate_no_trade = False

            try:
                s_min = int(getattr(config, 'micro_short_min_ticks_1m', 12))
                std_eps = float(getattr(config, 'gate_std_short_epsilon', 1e-6))
                tight_thr = float(getattr(config, 'gate_tightness_threshold', 0.995))
                gate_no_trade = (n_short < s_min) or ((std_short <= std_eps) and (tightness >= tight_thr))
            except Exception:
                pass

            logger.debug(f"[{name}] Gate: n_short={n_short} std_short={std_short:.6f} tight={tightness:.3f} → no_trade={gate_no_trade}")





            ema_fast = float(ema_feats.get('ema_8', 0.0))
            ema_slow = float(ema_feats.get('ema_21', 0.0))
            ema_trend = 1.0 if ema_fast > ema_slow else (-1.0 if ema_fast < ema_slow else 0.0)

            micro_slope = float(micro_ctx.get('slope', 0.0))
            imbalance = float(micro_ctx.get('imbalance', 0.0))
            mean_drift_pct = float(micro_ctx.get('vwap_drift_pct', micro_ctx.get('mean_drift_pct', 0.0)))
            mean_drift = mean_drift_pct / 100.0

            enable_rule_blend = bool(getattr(config, 'enable_rule_blend', True))
            indicator_score = None
            if enable_rule_blend:
                indicator_features = {
                    'ema_trend': ema_trend,
                    'micro_slope': micro_slope,
                    'imbalance': imbalance,
                    'mean_drift': mean_drift,
                }
                try:
                    indicator_score = model_pipe.compute_indicator_score(indicator_features)
                    logger.debug(f"[{name}] Indicator score: {indicator_score:.3f}")
                except Exception as e:
                    logger.debug(f"[{name}] Indicator score computation failed: {e}")

            # Scale by recent tick volatility
            try:
                px = np.array(prices[-64:], dtype=float)
                if px.size >= 3:
                    ret_std = float(np.std(np.diff(px)))
                    scale = max(1e-6, ret_std)
                else:
                    scale = 1.0
            except Exception as e:
                logger.debug(f"[{name}] Scale computation failed: {e}")
                scale = 1.0

            features_raw = {
                **ema_feats,
                **ofd,
                'micro_slope': micro_slope,
                'micro_imbalance': imbalance,
                'mean_drift_pct': mean_drift_pct,
                'last_price': tick.get('ltp', 0.0),
            }
            features = FeaturePipeline.normalize_features(features_raw, scale=scale)

            # Live tensor
            try:
                live_tensor = ws_handler.get_live_tensor() if hasattr(ws_handler, 'get_live_tensor') else _build_live_tensor_from_prices(prices, 64)
            except Exception:
                live_tensor = _build_live_tensor_from_prices(prices, 64)

            # Build staged row for training (features + latent used for this minute's call)
            try:
                latent_for_log = None
                try:
                    latent_for_log = model_pipe.cnn_lstm.predict(live_tensor)
                    latent_for_log = np.atleast_1d(np.asarray(latent_for_log, dtype=float)).ravel().tolist()
                except Exception:
                    latent_for_log = None
                
                # Keep the dict to preserve feature names/order
                features_for_log = dict(features)
            except Exception:
                features_for_log = {}
                latent_for_log = None

            # Profit factor / RL context
            try:
                recent_pf = ws_handler.get_recent_profit_factor() if hasattr(ws_handler, "get_recent_profit_factor") else 1.0
            except Exception:
                recent_pf = 1.0

            # Predict
            try:
                signal_probs, alpha_buy = model_pipe.predict(
                    live_tensor=live_tensor,
                    engineered_features=list(features.values()),
                    recent_profit_factor=recent_pf,
                    indicator_score=indicator_score if enable_rule_blend else None,
                )
                buy_prob = float(signal_probs[0][1])
                logger.debug(f"[{name}] Prediction: buy_prob={buy_prob:.3f}, alpha={float(alpha_buy):.3f}")
            except Exception as e:
                logger.error(f"[{name}] Model inference error: {e}", exc_info=True)
                signal_probs, alpha_buy = (np.array([[0.5, 0.5]]), 0.6)
                buy_prob = 0.5

            # Cache last-minute prediction
            try:
                decision = _make_decision(buy_prob, float(alpha_buy))
                last_minute_prediction = {
                    "buy_prob": buy_prob,
                    "alpha": float(alpha_buy),
                    "decision": decision,
                    "ltp": float(tick.get('ltp', 0.0)),
                    "no_trade": bool(gate_no_trade)
                }
            except Exception:
                pass

            # Hold staged info for the current bucket (used at rollover)
            staged_row_current_bucket = {
                "features": features_for_log,
                "latent": latent_for_log,
                "indicator_score": float(indicator_score) if indicator_score is not None else None,
                "decision": decision,
                "buy_prob": float(buy_prob),
                "alpha": float(alpha_buy),
                "no_trade": bool(gate_no_trade),
                "ltp": float(tick.get('ltp', 0.0)),
            }

            # Execution decision
            try:
                best_px = ws_handler.get_best_price() if hasattr(ws_handler, 'get_best_price') else float(tick.get('ltp', 0.0))
                mid_px = ws_handler.get_mid_price() if hasattr(ws_handler, 'get_mid_price') else _extract_best_and_mid_from_tick(tick)[1]
            except Exception:
                best_px, mid_px = _extract_best_and_mid_from_tick(tick)

            try:
                fill_prob = ws_handler.get_fill_prob() if hasattr(ws_handler, 'get_fill_prob') else 0.5
                time_waited = ws_handler.get_time_waited() if hasattr(ws_handler, 'get_time_waited') else 0.0
            except Exception:
                fill_prob, time_waited = 0.5, 0.0

            try:
                target_price = feat_pipe.place_order(
                    price=best_px,
                    fill_prob=fill_prob,
                    time_waited=time_waited,
                    get_mid_price_func=lambda: mid_px,
                )
                logger.debug(
                    f"[{name}] Order decision: target_price={float(target_price):.2f}, "
                    f"alpha={float(alpha_buy):.3f}, buy_prob={buy_prob:.3f}"
                )
            except Exception as e:
                logger.debug(f"[{name}] Order placement skipped: {e}")

            # Drift detection and alert (throttled)
            try:
                drift_stats = feat_pipe.detect_drift(features)

                alert_needed = any(stat.get('ks_stat', 0.0) > 0.2 for stat in drift_stats.values())
                if alert_needed:
                    logger.info(f"[{name}] Drift detected (ks>0.2) on one or more features")

                
            except Exception as e:
                logger.debug(f"[{name}] Drift detection skipped: {e}")

            await asyncio.sleep(0)

    except asyncio.CancelledError:
        logger.info(f"[{name}] Processing loop cancelled")
    except Exception as e:
        logger.error(f"[{name}] Fatal processing error: {e}", exc_info=True)
    finally:
        logger.info(f"[{name}] Processing loop shutdown")


# ------------------------- Main Orchestrator ------------------------- #

async def main_loop(config, cnn_lstm, xgb, rl_agent, train_features, token_b64, chat_id):
    """
    Orchestrates:
    - Multiple WS connections (each: connector + heartbeat + processing tasks).
    - Global feature/model pipeline and weight refresh loop.
    - Graceful shutdown of all tasks.
    - NEW: Feature logging, PF tracking, fair HOLD/FLAT scoring, no-trade gate

    Accepts a single config or config.connections (list/dict) for multiple feeds.
    """
    logger.info("=" * 60)
    logger.info("STARTING MAIN EVENT LOOP")
    logger.info("=" * 60)

    connections = _normalize_connections(config)
    logger.info(f"Total connections configured: {len(connections)}")

    # Shared components
    try:
        feat_pipe = FeaturePipeline(train_features=train_features, rl_agent=rl_agent)
        model_pipe = create_default_pipeline(cnn_lstm, xgb, rl_agent)
        telegram = TelegramBot(token_b64, chat_id)
        logger.info("Global components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize global components: {e}", exc_info=True)
        raise

    # Global weight refresh
    hitrate_path = getattr(config, 'hitrate_path', 'hitrate.txt')
    weight_refresh_seconds = int(getattr(config, 'weight_refresh_seconds', 30))
    weight_task = asyncio.create_task(weight_refresh_loop(model_pipe, hitrate_path, weight_refresh_seconds))
    logger.info(f"Weight refresh task started (interval: {weight_refresh_seconds}s)")

    # NEW: Feature log path
    feature_log_path = getattr(config, 'feature_log_path', 'feature_log.csv')
    logger.info(f"Feature log path: {feature_log_path}")

    # Per-connection tasks
    stop_events: Dict[str, asyncio.Event] = {}
    ws_tasks = []
    hb_tasks = []
    proc_tasks = []

    for name, cfg in connections:
        try:
            ws_handler = WSHandler(cfg)
            _install_preclose_shim_if_missing(ws_handler, cfg)

            # Per-connection tick queue and OB ring
            tick_queue: asyncio.Queue = asyncio.Queue()
            ob_ring = deque(maxlen=10)

            # EVAL registry per connection
            pending_preds: Dict[datetime, Dict[str, float]] = {}

            def _write_hitrate_records(candle_ts: datetime, hit: int):
                """
                Write one JSONL record per tunable indicator so weight tuner can adapt.
                """
                try:
                    indicators = ['micro_slope', 'imbalance', 'mean_drift']
                    with open(getattr(cfg, 'hitrate_path', 'hitrate.txt'), 'a') as f:
                        for ind in indicators:
                            rec = {
                                "timestamp": candle_ts.isoformat(),
                                "indicator": ind,
                                "hit": int(hit),
                                "miss": int(1 - hit)
                            }
                            f.write(json.dumps(rec) + "\n")
                    logger.info(f"[{name}] [EVAL] Wrote hitrate for {candle_ts.strftime('%H:%M:%S')}: hit={hit}")
                except Exception as e:
                    logger.warning(f"[{name}] [EVAL] Failed to write hitrate: {e}")

            # Async callbacks
            async def _on_tick_cb(tick: Dict[str, Any]):
                try:
                    best_px, mid_px = _extract_best_and_mid_from_tick(tick)
                    ob_snapshot = {
                        'bid_price': best_px if best_px <= mid_px else mid_px,
                        'ask_price': best_px if best_px >= mid_px else mid_px
                    }
                    ob_ring.append(ob_snapshot)
                    await tick_queue.put(tick)
                except Exception as e:
                    logger.debug(f"[{name}] on_tick callback error (ignored): {e}")

            # NEW: Enhanced candle callback with fair HOLD/FLAT scoring and feature logging
            async def _on_candle_cb(candle_df, full_df):
                try:
                    idx_ts = candle_df.index[-1] if not candle_df.empty else None
                    if idx_ts is not None:
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
                        
                        # EVAL: score the just-closed candle if we had a registered prediction
                        try:
                            # Access pending_train_rows from processing loop context
                            pred = pending_preds.pop(idx_ts, None)
                            staged = getattr(_on_candle_cb, '_pending_train_rows', {}).pop(idx_ts, None)
                            
                            if pred is not None:
                                o = float(row.get('open', 0.0))
                                c = float(row.get('close', 0.0))
                                move = c - o
                                direction = "UP" if c > o else ("DOWN" if c < o else "FLAT")
                                expected = "BUY" if pred["buy_prob"] >= pred["alpha"] else ("SELL" if (1.0 - pred["buy_prob"]) >= pred["alpha"] else "HOLD")
                                
                                # Fair HOLD/FLAT scoring
                                flat_tol = float(getattr(cfg, 'flat_tolerance_pct', 0.0002))  # 0.02% default
                                is_flat = abs(move) <= (flat_tol * max(1e-6, o))
                                skip_eval = bool(pred.get("no_trade", False))
                                
                                if skip_eval:
                                    logger.info(f"[{name}] [EVAL] {idx_ts.strftime('%H:%M:%S')} pred={expected} (bp={pred['buy_prob']:.3f}, thr={pred['alpha']:.3f}) "
                                                f"actual={direction} → SKIPPED (no-trade gate)")
                                    hit = None
                                else:
                                    if expected == "HOLD" and is_flat:
                                        hit = 1
                                    elif expected == "BUY" and c > o:
                                        hit = 1
                                    elif expected == "SELL" and c < o:
                                        hit = 1
                                    else:
                                        hit = 0
                                    
                                    logger.info(
                                        f"[{name}] [EVAL] {idx_ts.strftime('%H:%M:%S')} pred={expected} (bp={pred['buy_prob']:.3f}, thr={pred['alpha']:.3f}) "
                                        f"actual={direction} (O={o:.2f}→C={c:.2f}, is_flat={is_flat}) → {'HIT' if hit == 1 else 'MISS'}"
                                    )
                                    
                                    # Hit-rate logging only when scored
                                    if hit is not None:
                                        _write_hitrate_records(idx_ts, hit)
                                    
                                    # PF update (only for directional trades)
                                    if expected in ("BUY", "SELL"):
                                        pnl = (c - o) if expected == "BUY" else (o - c)
                                        try:
                                            ws_handler.record_pnl(pnl)
                                            logger.info(f"[{name}] [PF] Recorded PnL={pnl:.2f} after {expected}")
                                        except Exception:
                                            pass
                                
                                # Training row: write features + label
                                try:
                                    # Use staged row if available; otherwise minimal row
                                    feat_map = (staged or {}).get("features", {})
                                    latent = (staged or {}).get("latent", None)
                                    dec = (staged or pred).get("decision", expected)
                                    bp = float((staged or pred).get("buy_prob", 0.5))
                                    alpha = float((staged or pred).get("alpha", 0.6))
                                    no_trade = bool((staged or pred).get("no_trade", False))
                                    label = "BUY" if c > o else ("SELL" if c < o else "FLAT")
                                    tradeable = not no_trade
                                    
                                    # Append CSV
                                    cols = [
                                        idx_ts.isoformat(), dec, label, f"{bp:.6f}", f"{alpha:.6f}",
                                        str(tradeable), str(is_flat), str(int(row.get('tick_count', 0)))
                                    ]
                                    # features (name=value)
                                    for k, v in feat_map.items():
                                        cols.append(f"{k}={float(v):.8f}")
                                    # latent
                                    if latent is not None:
                                        cols.append("latent=" + "|".join(f"{float(x):.8f}" for x in latent))
                                    with open(feature_log_path, "a", encoding="utf-8") as f:
                                        f.write(",".join(cols) + "\n")
                                    logger.info(f"[{name}] [TRAIN] Logged features for {idx_ts.strftime('%H:%M:%S')} label={label} tradeable={tradeable}")
                                except Exception as e:
                                    logger.warning(f"[{name}] [TRAIN] Failed to log training row: {e}")
                        except Exception as e:
                            logger.warning(f"[{name}] [EVAL] Evaluation error: {e}")
                except Exception:
                    pass

            # Initialize pending_train_rows as callback attribute for cross-function access
            _on_candle_cb._pending_train_rows = {}

            async def _on_preclose_cb(preview_df, hist_df):
                try:
                    idx_ts = preview_df.index[-1] if not preview_df.empty else None
                    if idx_ts is not None:
                        row = preview_df.iloc[-1]
                        logger.info(
                            f"[{name}] Pre-close preview at {idx_ts.strftime('%H:%M:%S')} | "
                            f"O:{float(row.get('open', 0.0)):.2f} "
                            f"H:{float(row.get('high', 0.0)):.2f} "
                            f"L:{float(row.get('low', 0.0)):.2f} "
                            f"C:{float(row.get('close', 0.0)):.2f} "
                            f"Vol:{int(row.get('volume', 0)):,}"
                        )
                except Exception:
                    pass

            # Attach callbacks on handler
            try:
                ws_handler.on_tick = _on_tick_cb
                ws_handler.on_candle = _on_candle_cb
                ws_handler.on_preclose = _on_preclose_cb
            except Exception:
                pass

            # Start connector and heartbeat
            stop_event = asyncio.Event()
            stop_events[name] = stop_event
            ws_task = asyncio.create_task(_ws_connect_and_stream(name, cfg, ws_handler, stop_event))
            hb_task = asyncio.create_task(_websocket_heartbeat(name, ws_handler, interval_sec=30))
            ws_tasks.append((name, ws_handler, ws_task))
            hb_tasks.append((name, hb_task))
            logger.info(f"[{name}] Connector and heartbeat started")

            # Start processing loop with feature_log_path
            proc_task = asyncio.create_task(
                _connection_processing_loop(
                    name=name,
                    ws_handler=ws_handler,
                    feat_pipe=feat_pipe,
                    model_pipe=model_pipe,
                    telegram=telegram,
                    config=cfg,
                    tick_queue=tick_queue,
                    ob_ring=ob_ring,
                    pending_preds=pending_preds,
                    feature_log_path=feature_log_path,
                    pending_train_rows=_on_candle_cb._pending_train_rows  # Pass reference
                )
            )
            proc_tasks.append((name, proc_task))
            logger.info(f"[{name}] Processing task started")

        except Exception as e:
            logger.error(f"[{name}] Failed to initialize connection: {e}", exc_info=True)

    if not ws_tasks:
        logger.critical("No connections could be initialized. Exiting.")
        try:
            weight_task.cancel()
            await asyncio.wait_for(weight_task, timeout=5.0)
        except Exception:
            pass
        return

    # Keep main alive
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Main event loop cancelled")
    except Exception as e:
        logger.error(f"Fatal error in main_loop: {e}", exc_info=True)
    finally:
        # Shutdown sequence
        logger.info("Shutting down main loop")

        # Signal connectors to stop
        for name, _handler, _task in ws_tasks:
            try:
                stop_events[name].set()
            except Exception:
                pass

        # Cancel connector tasks
        for name, _handler, ws_task in ws_tasks:
            try:
                ws_task.cancel()
                await asyncio.wait_for(ws_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"[{name}] Error cancelling connector task: {e}")

        # Cancel heartbeats
        for name, hb_task in hb_tasks:
            try:
                hb_task.cancel()
                await asyncio.wait_for(hb_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"[{name}] Error cancelling heartbeat: {e}")

        # Cancel processing loops
        for name, proc_task in proc_tasks:
            try:
                proc_task.cancel()
                await asyncio.wait_for(proc_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(f"[{name}] Error cancelling processing: {e}")

        # Cancel weight refresh
        try:
            weight_task.cancel()
            await asyncio.wait_for(weight_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.error(f"Error cancelling weight task: {e}")

        logger.info("=" * 60)
        logger.info("MAIN EVENT LOOP SHUTDOWN COMPLETE")
        logger.info("=" * 60)
