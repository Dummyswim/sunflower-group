"""
DhanHQ v2 WebSocket handler - Enhanced (volume-free, no synthetic volume)
Previously: contained synthetic volume generation (REMOVED per new requirements)
"""
import asyncio
import struct
import json
import logging
from typing import Dict, Optional, Any, List
import websockets
import pandas as pd
import numpy as np
import base64

logger = logging.getLogger(__name__)

from datetime import datetime, timezone, timedelta 
IST = timezone(timedelta(hours=5, minutes=30))

class EnhancedWebSocketHandler:
    """WebSocket handler for DhanHQ v2 — volume-free."""

    # DhanHQ v2 Packet Types (retained for compatibility)
    PACKET_TYPES = {
        8: "ticker",
        16: "quote",
        32: "index_full",
        44: "equity_full",
        50: "quote_extended",
        66: "quote_full",
        162: "full_packet",
        184: "market_depth",
        492: "market_depth_50"
    }
    
    # Response codes from websocket_client.py (unchanged)
    TICKER_PACKET = 2
    QUOTE_PACKET = 4
    OI_PACKET = 5
    PREV_CLOSE_PACKET = 6
    MARKET_STATUS_PACKET = 7
    FULL_PACKET = 8
    DISCONNECT_PACKET = 50
    
    def __init__(self, config):
        logger.info("Initializing Enhanced WebSocket Handler (volume-free)")
        self.config = config
        self.websocket = None
        self.authenticated = False
        self.running = True
        
        # Data buffers
        self.tick_buffer = []
        self.candle_data = pd.DataFrame()
        self.current_candle = {
            'ticks': [],
            'start_time': None
        }
        
        # REMOVED: Synthetic volume tracking and fields (last_price/current_period_volume/base etc.) [[20]]
        # self.last_price = None
        # self.last_cumulative_volume = 0
        # self.current_period_volume = 0
        # self.synthetic_volume_base = 1000

        self.current_oi = None
        self._last_candle_oi = None
        logger.debug("OI tracking initialized: current_oi=None, _last_candle_oi=None")

        # Callbacks 
        self.on_tick = None 
        self.on_candle = None 
        self.on_error = None 
        self.on_preclose = None 
        
        # Microstructure buffers
        from collections import deque
        win = int(getattr(self.config, 'micro_tick_window', 400))
        self._micro_ticks = deque(maxlen=max(50, win))
        logger.info("[MICRO] tick window size set to %d", self._micro_ticks.maxlen)

        self._micro_last_snapshot = {}
        logger.debug("Microstructure buffers initialized")

        # NEW: Async queue for external consumers (compatibility with main_event_loop.get_next_tick) [[15]]
        self._tick_queue: asyncio.Queue = asyncio.Queue()
        self._last_best = None  # track last best price/mid for getters
        self._last_depth = None # track last market depth snapshot

        # Statistics
        self.packet_stats = {packet_type: 0 for packet_type in self.PACKET_TYPES.values()}
        self.packet_stats.update({'ticker': 0, 'quote': 0, 'full': 0, 'oi': 0, 'other': 0})
        self.last_packet_time = None
        self.tick_count = 0
        self._diag_ticks_left = 50
        
        self.boundary_task = None
        self.data_watchdog_task = None
        self._last_subscribe_time = None

        # Optional checksum validation
        self.enable_packet_checksum_validation = bool(getattr(self.config, 'enable_packet_checksum_validation', False))
        self._checksum_warned = False
        self._checksum_mismatch_count = 0
        logger.debug(f"Checksum validation enabled: {self.enable_packet_checksum_validation}")

        # Pre-close and boundary close state 
        self._preclose_fired_for_bucket = None
        self._preclose_lock = asyncio.Lock()

        self._bucket_closed = False
        
        logger.info(f"Configuration: SecurityId={config.nifty_security_id}, "
                   f"Interval={config.candle_interval_seconds}s, "
                   f"MaxBuffer={config.max_buffer_size}")
    

    def _normalize_tick_ts(self, ltt: int) -> datetime:
        """
        Normalize exchange timestamp to IST.
        """
        now_ist = datetime.now(IST)
        ts_utc_to_ist = datetime.fromtimestamp(ltt, tz=timezone.utc).astimezone(IST)
        ts_direct_ist = datetime.fromtimestamp(ltt, tz=IST)
        if abs((now_ist - ts_utc_to_ist).total_seconds()) <= abs((now_ist - ts_direct_ist).total_seconds()):
            return ts_utc_to_ist
        return ts_direct_ist
    

    def get_ticks_between(self, start_ts, end_ts) -> list:
        """Return a list of tick dicts with timestamps in [start_ts, end_ts)."""
        try:
            if not self.tick_buffer:
                return []
            out = [t for t in self.tick_buffer if t.get('timestamp') and (start_ts <= t['timestamp'] < end_ts)]
            logger.info("[TICKS] window %s→%s count=%d", start_ts.strftime('%H:%M:%S'), end_ts.strftime('%H:%M:%S'), len(out))
            return out
        except Exception as e:
            logger.debug(f"get_ticks_between error: {e}")
            return []

    def _assemble_candle(self, start_time: datetime, ticks: List[Dict]) -> pd.DataFrame: 
        """Build an OHLC candle from accumulated ticks for this bucket (volume-free)."""
        try: 
            prices = [t.get('ltp', 0.0) for t in ticks if t.get('ltp', 0.0) > 0] 
            if not prices: 
                return pd.DataFrame() 
            open_price = float(prices[0]) 
            high = float(max(prices)) 
            low = float(min(prices)) 
            close = float(prices[-1])

            # REMOVED: volume calculation/storage (synthetic/real) [[20]]
            # candle_volume = ...

            # OI integration (safe defaults)
            try:
                oi_val = int(self.current_oi) if isinstance(self.current_oi, (int, float)) else int(self._last_candle_oi) if self._last_candle_oi is not None else 0
            except Exception:
                oi_val = 0

            try:
                prev_oi = int(self.candle_data['oi'].iloc[-1]) if ('oi' in self.candle_data.columns and not self.candle_data.empty) else oi_val
            except Exception:
                prev_oi = oi_val

            oi_change = int(oi_val - prev_oi)
            denom = abs(prev_oi) if prev_oi != 0 else 1
            oi_change_pct = float((oi_change / denom) * 100.0)

            logger.info(f"[CANDLE-PREVIEW] OI={oi_val} ΔOI={oi_change} ({oi_change_pct:.2f}%)")
            
            # CHANGED: No 'volume' column in candle DataFrame (volume-free)
            return pd.DataFrame([{
                'timestamp': start_time,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'tick_count': len(ticks),
                'oi': oi_val,
                'oi_change': oi_change,
                'oi_change_pct': oi_change_pct
            }]).set_index('timestamp')

        except Exception as e:
            logger.debug(f"_assemble_candle error: {e}")
            return pd.DataFrame()

    async def _maybe_fire_preclose(self, now_ts: datetime): 
        """Trigger a pre-close preview once per bucket near its end."""
        try: 
            if not self.on_preclose or not self.current_candle['start_time']: 
                return 
            start = self.current_candle['start_time'] 
            interval_min = max(1, self.config.candle_interval_seconds // 60) 
            close_time = start + timedelta(minutes=interval_min) 
            lead = getattr(self.config, 'preclose_lead_seconds', 10)
            preclose_time = close_time - timedelta(seconds=lead)  # ADD THIS LINE

            has_ticks = bool(self.current_candle.get('ticks'))
            fired = (self._preclose_fired_for_bucket == start)

            if now_ts < preclose_time:


                return
            if fired:
                logger.debug(
                    f"[Pre-Close] Skip: already fired for bucket start={start.strftime('%H:%M:%S')}"
                )
                return
            if not has_ticks:
                logger.info(
                    f"[Pre-Close] Skip: no ticks in current bucket (start={start.strftime('%H:%M:%S')})"
                )
                return

            async with self._preclose_lock:
                if self._preclose_fired_for_bucket == start:
                    logger.debug(f"[Pre-Close] Skip inside lock: already fired for start={start.strftime('%H:%M:%S')}")
                    return

                preview = self._assemble_candle(start, self.current_candle['ticks'])
                if not preview.empty:
                    self._preclose_fired_for_bucket = start
                    logger.info("=" * 60)
                    logger.info("⏳ Pre-close checkpoint: start=%s close=%s fired_at=%s",
                                start.strftime('%H:%M:%S'), close_time.strftime('%H:%M:%S'), now_ts.strftime('%H:%M:%S'))
                    logger.info("=" * 60)
                    await self.on_preclose(
                        preview,
                        self.candle_data.copy() if not self.candle_data.empty else preview.copy()
                    )
        except Exception as e:
            logger.debug(f"Pre-close skipped: {e}")











    async def _boundary_close_loop(self): 
        """Close the current bucket at the time boundary without waiting for the next tick."""
        logger.info("Starting boundary close loop") 
        while self.running: 
            try: 
                await asyncio.sleep(1.0) 
                iteration_count = getattr(self, '_boundary_iterations', 0)
                if iteration_count > 86400:
                    logger.warning("Boundary loop max iterations reached, resetting")
                    self._boundary_iterations = 0
                else:
                    self._boundary_iterations = iteration_count + 1
                now_ts = datetime.now(IST) 
                start = self.current_candle.get('start_time')
                if int(now_ts.timestamp()) % 30 == 0:
                    logger.debug(f"Boundary loop alive at {now_ts.strftime('%H:%M:%S')}")
                if start and self.current_candle.get('ticks'):
                    close_time = start + timedelta(seconds=self.config.candle_interval_seconds)
                    if now_ts >= close_time and not self._bucket_closed:
                        buf = int(getattr(self.config, 'preclose_completion_buffer_sec', 1))
                        if buf > 0 and (now_ts - close_time).total_seconds() < (buf + 5):
                            logger.info("[PRECLOSE] Finalize buffer: sleeping %ds before closing bucket", buf)
                            await asyncio.sleep(buf)
                        logger.info("=" * 60)
                        logger.info("⏱️ Boundary close %s→%s — creating candle & dispatching callbacks",
                                    start.strftime('%H:%M:%S'), close_time.strftime('%H:%M:%S'))
                        logger.info("=" * 60)
                        await self._create_candle(start, self.current_candle['ticks'])
                        self._bucket_closed = True
            except Exception as e: 
                logger.error(f"Boundary close loop error: {e}", exc_info=True) 
                await asyncio.sleep(1.0)

    def get_micro_features(self) -> Dict[str, float]:
        """
        Compute microstructure features over DUAL time-bounded windows (short + long).
        Returns dict; NaN/Inf-safe; volume-free.
        """
        from datetime import datetime, timedelta
        out = {'imbalance': 0.0, 'slope': 0.0, 'std_dltp': 0.0, 'mean_drift_pct': 0.0, 'n': 0,  # CHANGED: vwap_drift_pct→mean_drift_pct [[20]]
            'imbalance_short': 0.0, 'slope_short': 0.0, 'std_dltp_short': 0.0, 
            'n_short': 0, 'momentum_delta': 0.0}
        try:
            if not self._micro_ticks:
                return out
            
            win_sec = int(getattr(self.config, 'micro_window_sec_1m', 25))
            min_ticks = int(getattr(self.config, 'micro_min_ticks_1m', 30))
            cutoff = datetime.now(IST) - timedelta(seconds=win_sec)
            recent = [(ts, px) for (ts, px) in list(self._micro_ticks) if ts >= cutoff]
            
            s_sec = int(getattr(self.config, 'micro_short_window_sec_1m', 8))
            s_min = int(getattr(self.config, 'micro_short_min_ticks_1m', 12))
            cutoff_s = datetime.now(IST) - timedelta(seconds=s_sec)
            recent_s = [(ts, px) for (ts, px) in recent if ts >= cutoff_s]
            
            if len(recent) < min_ticks:
                out['n'] = len(recent)
                out['n_short'] = len(recent_s)
                return out
            
            ts, px = zip(*recent)
            n = len(px); out['n'] = n
            px_arr = np.asarray(px, dtype=float)
            d = np.diff(px_arr)
            d = d[~np.isclose(d, 0.0)]
            out['std_dltp'] = float(np.nan_to_num(np.std(d))) if d.size else 0.0
            ups = int(np.sum(d > 0)); dns = int(np.sum(d < 0)); total = ups + dns
            out['imbalance'] = float((ups - dns) / max(1, total)) if total > 0 else 0.0
            out['moves_up'] = int(ups)
            out['moves_dn'] = int(dns)
            out['moves_total'] = int(total)

            x = np.arange(n, dtype=float)
            x = (x - x.mean()) / max(1e-9, x.std())
            y = (px_arr - px_arr.mean()) / max(1e-9, px_arr.std())
            w = np.linspace(0.5, 1.0, n)
            beta = float(np.dot(w * x, y) / max(1e-9, np.dot(w * x, x)))
            out['slope'] = beta

            try:
                hi_long = float(np.max(px_arr))
                lo_long = float(np.min(px_arr))
                last_px = float(px_arr[-1])
                rng = max(1e-9, hi_long - lo_long)
                micro_pos = float((last_px - lo_long) / rng)
                out['hi_long'] = hi_long
                out['lo_long'] = lo_long
                out['micro_pos'] = micro_pos
            except Exception:
                out['hi_long'] = out['lo_long'] = None
                out['micro_pos'] = 0.5

            try:
                s = pd.Series(px_arr, dtype='float64')
                ema20 = float(s.ewm(span=min(20, len(s)), adjust=False).mean().iloc[-1])
                ema50 = float(s.ewm(span=min(50, len(s)), adjust=False).mean().iloc[-1])
                out['ema20_1m'] = ema20
                out['ema50_1m'] = ema50
                out['price_above_ema20'] = bool(last_px > ema20)
            except Exception:
                out['ema20_1m'] = out['ema50_1m'] = 0.0
                out['price_above_ema20'] = False

            try:
                k_sec = int(getattr(self.config, 'micro_last3s_window_sec_1m', 3))
                cutoff_k = datetime.now(IST) - timedelta(seconds=k_sec)
                recent_k = [(ts, px) for (ts, px) in recent if ts >= cutoff_k]
                if len(recent_k) >= 6:
                    _, px_k = zip(*recent_k)
                    arrk = np.asarray(px_k, dtype=float)
                    dk = np.diff(arrk)
                    dk = dk[~np.isclose(dk, 0.0)]
                    ups_k = int(np.sum(dk > 0))
                    dns_k = int(np.sum(dk < 0))
                    tot_k = ups_k + dns_k
                    out['last3s_imbalance'] = float((ups_k - dns_k) / max(1, tot_k)) if tot_k > 0 else 0.0
                    xk = np.arange(len(arrk), dtype=float)
                    xk = (xk - xk.mean()) / max(1e-9, xk.std())
                    yk = (arrk - arrk.mean()) / max(1e-9, arrk.std())
                    wk = np.linspace(0.6, 1.0, len(arrk))
                    out['last3s_slope'] = float(np.dot(wk * xk, yk) / max(1e-9, np.dot(wk * xk, xk)))
                else:
                    out['last3s_imbalance'] = 0.0
                    out['last3s_slope'] = 0.0
            except Exception as e:
                out['last3s_imbalance'] = 0.0
                out['last3s_slope'] = 0.0

            logger.info("[MICRO-CTX] micro_pos=%.2f | ema20=%.2f ema50=%.2f | last3s slp=%.3f imb=%.2f",
                        out.get('micro_pos', 0.5),
                        out.get('ema20_1m', 0.0),
                        out.get('ema50_1m', 0.0),
                        out.get('last3s_slope', 0.0),
                        out.get('last3s_imbalance', 0.0))

            # CHANGED: mean drift vs arithmetic mean (no VWAP) [[20]]
            mean_px = float(np.mean(px_arr))
            last = float(px_arr[-1])
            out['mean_drift_pct'] = float(((last - mean_px) / max(1e-9, mean_px)) * 100.0)
            
            out['n_short'] = len(recent_s)
            if len(recent_s) >= s_min:
                ts_s, px_s = zip(*recent_s)
                pxs = np.asarray(px_s, dtype=float)
                d_s = np.diff(pxs); d_s = d_s[~np.isclose(d_s, 0.0)]
                out['std_dltp_short'] = float(np.nan_to_num(np.std(d_s))) if d_s.size else 0.0
                ups_s = int(np.sum(d_s > 0)); dns_s = int(np.sum(d_s < 0)); tot_s = ups_s + dns_s
                out['imbalance_short'] = float((ups_s - dns_s) / max(1, tot_s)) if tot_s > 0 else 0.0
                x_s = np.arange(len(pxs), dtype=float)
                x_s = (x_s - x_s.mean()) / max(1e-9, x_s.std())
                y_s = (pxs - pxs.mean()) / max(1e-9, pxs.std())
                w_s = np.linspace(0.6, 1.0, len(pxs))
                beta_s = float(np.dot(w_s * x_s, y_s) / max(1e-9, np.dot(w_s * x_s, x_s)))
                out['slope_short'] = beta_s
            
            out['momentum_delta'] = float(out['slope_short'] - out['slope'])
            
            if not np.isfinite(out['std_dltp']) or out['std_dltp'] < 0:
                out['std_dltp'] = 0.0
            if out['std_dltp'] > 50.0:
                logger.warning("[MICRO-GUARD] stdΔ outlier detected (%.4f) → clamped", out['std_dltp'])
                out['std_dltp'] = 0.0
            
            if not np.isfinite(out['mean_drift_pct']):
                out['mean_drift_pct'] = 0.0
            if abs(out['mean_drift_pct']) > 2.0:
                logger.warning("[MICRO-GUARD] drift%% outlier detected (%.4f%%) → clamped", out['mean_drift_pct'])
                out['mean_drift_pct'] = 2.0 if out['mean_drift_pct'] > 0 else -2.0

            logger.info("[MICRO] n=%d|%d (long|short) | moves=%d↑/%d↓ | imbL=%.2f imbS=%.2f | slpL=%.3f slpS=%.3f | stdL=%.5f stdS=%.5f | drift=%.3f%%",
                        out['n'], out['n_short'], out.get('moves_up', 0), out.get('moves_dn', 0),
                        out['imbalance'], out['imbalance_short'], out['slope'], out['slope_short'],
                        out['std_dltp'], out['std_dltp_short'], out['mean_drift_pct'])
        except Exception as e:
            logger.debug(f"[MICRO] snapshot error: {e}")
        return out

    # REMOVED: _calculate_synthetic_volume method (no synthetic/real volume used) [[20]]
    # def _calculate_synthetic_volume(self, ltp: float) -> int: ...

    async def connect(self):
        """Establish WebSocket connection with retry logic."""
        logger.info("Starting WebSocket connection to DhanHQ v2")
        max_attempts = self.config.max_reconnect_attempts
        attempt = 0

        # Reset per-connection state
        self.tick_buffer.clear()
        self.candle_data = pd.DataFrame()
        self.current_candle = {'ticks': [], 'start_time': None}
        # REMOVED: self.current_period_volume reset (volume-free) [[20]]
        # self.current_period_volume = 0
        self._preclose_fired_for_bucket = None
        self._bucket_closed = False
        logger.debug("Per-connection state reset")

        while attempt < max_attempts and self.running:
            try:
                attempt += 1
                logger.info(f"Connection attempt {attempt}/{max_attempts}")
                access_token = base64.b64decode(self.config.dhan_access_token_b64).decode("utf-8")
                client_id = base64.b64decode(self.config.dhan_client_id_b64).decode("utf-8")
                logger.debug(f"Credentials decoded for client: {client_id[:4]}****")
                ws_url = (
                    f"wss://api-feed.dhan.co?version=2"
                    f"&token={access_token}"
                    f"&clientId={client_id}"
                    f"&authType=2"
                )
                self.websocket = await websockets.connect(
                    ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,
                    compression=None
                )
                self.authenticated = True
                logger.info("WebSocket connected successfully")
                self.last_packet_time = None
                self._last_subscribe_time = None
                logger.debug("Per-connection timing reset")

                await self.subscribe()

                logger.info("WebSocket connection established and subscribed")
                if not self.boundary_task or self.boundary_task.done():
                    self.boundary_task = asyncio.create_task(self._boundary_close_loop())
                    self.boundary_task.add_done_callback(self._handle_task_exception)
                if not self.data_watchdog_task or self.data_watchdog_task.done():
                    self.data_watchdog_task = asyncio.create_task(self._data_watchdog_loop())
                    self.data_watchdog_task.add_done_callback(self._handle_task_exception)
                return True
            except Exception as e:
                logger.error(f"Connection attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    delay = min(self.config.reconnect_delay_base * (2 ** (attempt - 1)), 60)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        logger.error("Failed to establish WebSocket connection after all attempts")
        return False
    
    def _handle_task_exception(self, task):
        """Handle exceptions from background tasks."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Background task error: {e}", exc_info=True)
            
    async def subscribe(self):
        """Subscribe to NIFTY50 market data feed."""
        logger.info("Subscribing to NIFTY50 market data")
        try:
            subscription = {
                "RequestCode": 15,
                "InstrumentCount": 1,
                "InstrumentList": [{
                    "ExchangeSegment": self.config.nifty_exchange_segment,
                    "SecurityId": str(self.config.nifty_security_id)
                }]
            }  
            logger.info(f"[Subscribe] Sending subscription at {datetime.now(IST).strftime('%H:%M:%S')} with params: "
                        f"RequestCode={subscription['RequestCode']} "
                        f"ExchangeSegment={subscription['InstrumentList'][0]['ExchangeSegment']} "
                        f"SecurityId={subscription['InstrumentList'][0]['SecurityId']}")
            logger.info(f"[Subscribe] subscription: {subscription}")
            await self.websocket.send(json.dumps(subscription))
            self._last_subscribe_time = datetime.now(IST)
            logger.info(f"[Subscribe] _last_subscribe_time set to {self._last_subscribe_time.strftime('%H:%M:%S')}")
            logger.info("[Subscribe] Waiting for market data...")
            await asyncio.sleep(2)
            if self.tick_count == 0:
                logger.warning("[Subscribe] No ticks received after subscription - checking market status")
                status_request = {"RequestCode": 7}
                await self.websocket.send(json.dumps(status_request))
                logger.info("[Subscribe] Sent market status request")
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            raise
    
    def _parse_ticker_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Ticker packet (16 bytes) - volume-free."""
        if len(data) < 16:
            return None
        if getattr(self, 'enable_packet_checksum_validation', False):
            try:
                computed = (sum(data[:15]) & 0xFF)
                claimed = data[15]
                if computed != claimed:
                    self._checksum_mismatch_count += 1
                    if not self._checksum_warned or (self._checksum_mismatch_count % 100 == 0):
                        logger.warning(f"[Ticker] Checksum mismatch (computed={computed}, claimed={claimed}) — proceeding without drop")
                        self._checksum_warned = True
            except Exception as _e:
                logger.debug(f"[Ticker] Checksum validation skipped: {_e}")
        try:
            response_code = data[0]
            if response_code != self.TICKER_PACKET:
                return None
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id != self.config.nifty_security_id:
                return None
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
            ltt = struct.unpack('<I', data[12:16])[0]
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            try:
                ts = self._normalize_tick_ts(ltt)
                now_ist = datetime.now(IST)
                if abs((now_ist - ts).total_seconds()) > 3600 or ts.year < 2000:
                    timestamp = now_ist
                else:
                    timestamp = ts
            except Exception:
                timestamp = datetime.now(IST)

            # REMOVED: synthetic volume calculation and accumulation [[20]]
            self.packet_stats['ticker'] += 1
            logger.debug(f"Ticker: LTP={ltp:.2f}")

            # CHANGED: Track last best/mid for getters
            self._last_best = ltp

            return {
                'timestamp': timestamp,
                'packet_type': 'ticker',
                'ltp': ltp
            }
        except Exception as e:
            logger.error(f"Error parsing ticker packet: {e}")
            return None
    
    def _parse_quote_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Quote packet (50 or 66 bytes) - volume-free."""
        if len(data) < 50:
            return None
        try:
            response_code = data[0]
            if response_code != self.QUOTE_PACKET:
                return None
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id != self.config.nifty_security_id:
                return None
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
            ltq = struct.unpack('<h', data[12:14])[0]
            ltt = struct.unpack('<I', data[14:18])[0]
            atp = struct.unpack('<f', data[18:22])[0]
            volume = struct.unpack('<I', data[22:26])[0]  # present in packet but we will ignore it in outputs
            total_sell_qty = struct.unpack('<I', data[26:30])[0]
            total_buy_qty = struct.unpack('<I', data[30:34])[0]
            open_value = struct.unpack('<f', data[34:38])[0]
            close_value = struct.unpack('<f', data[38:42])[0]
            high_value = struct.unpack('<f', data[42:46])[0]
            low_value = struct.unpack('<f', data[46:50])[0]
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            try:
                ts = self._normalize_tick_ts(ltt)
                now_ist = datetime.now(IST)
                if abs((now_ist - ts).total_seconds()) > 3600 or ts.year < 2000:
                    timestamp = now_ist
                else:
                    timestamp = ts
            except Exception:
                timestamp = datetime.now(IST)

            # REMOVED: synthetic/real volume logic and accumulation [[20]]

            self.packet_stats['quote'] += 1
            if self.packet_stats['quote'] % 10 == 0:
                logger.info(f"Quote #{self.packet_stats['quote']}: LTP={ltp:.2f}, OHLC=[{open_value:.2f},{high_value:.2f},{low_value:.2f},{close_value:.2f}]")

            # CHANGED: Track last depth prices as simple best bid/ask snapshot if available later (depth packet)
            self._last_best = ltp

            return {
                'timestamp': timestamp,
                'packet_type': 'quote',
                'ltp': ltp,
                'ltq': ltq,
                'atp': atp,
                # REMOVED from output: 'volume'
                'total_sell_qty': total_sell_qty,  # retained raw values; we won't compute volume-based metrics with these
                'total_buy_qty': total_buy_qty,
                'open': open_value if open_value > 0 else ltp,
                'high': high_value if high_value > 0 else ltp,
                'low': low_value if low_value > 0 else ltp,
                'close': close_value if close_value > 0 else ltp
            }
        except Exception as e:
            logger.error(f"Error parsing quote packet: {e}")
            return None
    
    def _parse_full_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Full packet (162 bytes) with market depth - volume-free."""
        if len(data) < 162:
            return None
        try:
            response_code = data[0]
            if response_code != self.FULL_PACKET:
                return None
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id != self.config.nifty_security_id:
                return None
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
            ltq = struct.unpack('<h', data[12:14])[0]
            ltt = struct.unpack('<I', data[14:18])[0]
            atp = struct.unpack('<f', data[18:22])[0]
            volume = struct.unpack('<I', data[22:26])[0]  # ignored in outputs
            total_sell_qty = struct.unpack('<I', data[26:30])[0]
            total_buy_qty = struct.unpack('<I', data[30:34])[0]
            oi = struct.unpack('<I', data[34:38])[0]
            oi_high = struct.unpack('<I', data[38:42])[0]
            oi_low = struct.unpack('<I', data[42:46])[0]
            open_value = struct.unpack('<f', data[46:50])[0]
            close_value = struct.unpack('<f', data[50:54])[0]
            high_value = struct.unpack('<f', data[54:58])[0]
            low_value = struct.unpack('<f', data[58:62])[0]
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None

            # Parse market depth (5 levels)
            market_depth = []
            for i in range(5):
                start = 62 + (i * 20)
                bid_qty = struct.unpack('<I', data[start:start+4])[0]
                ask_qty = struct.unpack('<I', data[start+4:start+8])[0]
                bid_orders = struct.unpack('<H', data[start+8:start+10])[0]
                ask_orders = struct.unpack('<H', data[start+10:start+12])[0]
                bid_price = struct.unpack('<f', data[start+12:start+16])[0]
                ask_price = struct.unpack('<f', data[start+16:start+20])[0]
                market_depth.append({
                    'bid_qty': bid_qty,
                    'ask_qty': ask_qty,
                    'bid_orders': bid_orders,
                    'ask_orders': ask_orders,
                    'bid_price': bid_price,
                    'ask_price': ask_price
                })
            try:
                ts = self._normalize_tick_ts(ltt)
                now_ist = datetime.now(IST)
                if abs((now_ist - ts).total_seconds()) > 3600 or ts.year < 2000:
                    timestamp = now_ist
                else:
                    timestamp = ts
            except Exception:
                timestamp = datetime.now(IST)
            self.packet_stats['full'] += 1
            try:
                self.current_oi = int(oi)
                logger.debug(f"[OI] Updated from FULL packet: {self.current_oi}")
            except Exception:
                logger.debug("[OI] FULL packet OI update skipped (cast error)")
            logger.info(f"Full packet: LTP={ltp:.2f}, OI={oi}, Depth levels=5")

            # CHANGED: Store last depth snapshot for order book getters
            self._last_depth = market_depth
            self._last_best = ltp

            return {
                'timestamp': timestamp,
                'packet_type': 'full',
                'ltp': ltp,
                'ltq': ltq,
                'open': open_value if open_value > 0 else ltp,
                'high': high_value if high_value > 0 else ltp,
                'low': low_value if low_value > 0 else ltp,
                'close': close_value if close_value > 0 else ltp,
                'total_buy_qty': total_buy_qty,
                'total_sell_qty': total_sell_qty,
                'oi': oi,
                'market_depth': market_depth
            }
        except Exception as e:
            logger.error(f"Error parsing full packet: {e}")
            return None
    
    async def _process_tick(self, tick_data: Dict):
        """Process incoming tick data (volume-free)."""
        if not tick_data:
            return
        try:
            self.last_packet_time = datetime.now(IST)
            self.tick_count += 1

            # REMOVED: volume logging [[20]]
            self.tick_buffer.append(tick_data)

            # Microstructure tracking
            try:
                ts = tick_data.get('timestamp')
                ltp = float(tick_data.get('ltp', 0.0))
                if self._micro_ticks and ts and ts <= self._micro_ticks[-1][0]:
                    logger.debug("[MICRO] out-of-order tick skipped: %s <= %s", 
                                ts.strftime('%H:%M:%S') if hasattr(ts, 'strftime') else ts,
                                self._micro_ticks[-1][0].strftime('%H:%M:%S') if hasattr(self._micro_ticks[-1][0], 'strftime') else self._micro_ticks[-1][0])
                elif ts and ltp > 0:
                    self._micro_ticks.append((ts, ltp))
                    if len(self._micro_ticks) % 50 == 0:
                        logger.debug("[MICRO] buffer_len=%d", len(self._micro_ticks))
            except Exception as e:
                logger.debug("[MICRO] append error: %s", e)

            if len(self.tick_buffer) > self.config.max_buffer_size:
                self.tick_buffer.pop(0)
            
            current_time = tick_data['timestamp']
            interval_min = max(1, self.config.candle_interval_seconds // 60)
            bucket_min = (current_time.minute // interval_min) * interval_min 
            candle_start = current_time.replace(minute=bucket_min, second=0, microsecond=0)

            if self._diag_ticks_left > 0:
                self._diag_ticks_left -= 1
                logger.info(f"DBG: tick_ts={current_time.strftime('%H:%M:%S')} bucket={candle_start.strftime('%H:%M:%S')} "
                            f"start={self.current_candle['start_time'].strftime('%H:%M:%S') if self.current_candle['start_time'] else 'None'} "
                            f"ticks={len(self.current_candle['ticks']) if self.current_candle['ticks'] else 0}")

            if self.current_candle['start_time'] != candle_start: 
                if self.current_candle['start_time'] and self.current_candle['ticks'] and not self._bucket_closed: 
                    await self._create_candle(self.current_candle['start_time'], self.current_candle['ticks'])
                self.current_candle = {
                    'start_time': candle_start,
                    'ticks': [tick_data]
                }
                # REMOVED: self.current_period_volume reset [[20]]
                # self.current_period_volume = 0
                self._bucket_closed = False
                self._preclose_fired_for_bucket = None
                logger.debug(f"New candle period started: {candle_start.strftime('%H:%M:%S')}")
            else:
                self.current_candle['ticks'].append(tick_data)

            try:
                if (self.current_candle['start_time'] 
                    and (current_time - self.current_candle['start_time']).total_seconds() 
                        >= self.config.candle_interval_seconds 
                    and len(self.current_candle['ticks']) >= 3):
                    await self._create_candle(
                        self.current_candle['start_time'],
                        self.current_candle['ticks']
                    )
                    bucket_min = (current_time.minute // interval_min) * interval_min
                    self.current_candle = {
                        'start_time': current_time.replace(minute=bucket_min, second=0, microsecond=0),
                        'ticks': []
                    }
                    # REMOVED: current_period_volume reset [[20]]
                    # self.current_period_volume = 0
            except Exception as e:
                logger.debug(f"Safety flush skipped: {e}")

            try: 
                await self._maybe_fire_preclose(current_time) 
            except Exception as e: 
                logger.debug(f"Pre-close check failed: {e}")

            # NEW: push to async queue for get_next_tick [[15]]
            try:
                self._tick_queue.put_nowait((tick_data, self._last_depth, current_time))
            except Exception:
                pass

            if self.on_tick:
                await self.on_tick(tick_data)
        except Exception as e:
            logger.error(f"Tick processing error: {e}", exc_info=True)
    
    async def _create_candle(self, timestamp: datetime, ticks: List[Dict]):
        """Create OHLC candle (volume-free)."""
        if not ticks:
            return
        try:
            prices = [t['ltp'] for t in ticks if 'ltp' in t and t['ltp'] > 0]
            if not prices:
                logger.warning("No valid prices in ticks for candle creation")
                return
            open_price = prices[0]
            high = max(prices)
            low = min(prices)
            close = prices[-1]

            # REMOVED: all volume logic (synthetic/real) [[20]]

            hhmm = timestamp.hour * 100 + timestamp.minute 
            if timestamp.weekday() >= 5 or not (915 <= hhmm <= 1530): 
                logger.info(f"Skipping candle outside market hours: {timestamp.strftime('%H:%M:%S')}")
                return

            interval_min = max(1, self.config.candle_interval_seconds // 60) 
            candle_start = timestamp 

            try:
                oi_val = int(self.current_oi) if isinstance(self.current_oi, (int, float)) else int(self._last_candle_oi) if self._last_candle_oi is not None else 0
            except Exception:
                oi_val = 0
            try:
                prev_oi = int(self.candle_data['oi'].iloc[-1]) if ('oi' in self.candle_data.columns and not self.candle_data.empty) else oi_val
            except Exception:
                prev_oi = oi_val
            oi_change = int(oi_val - prev_oi)
            denom = abs(prev_oi) if prev_oi != 0 else 1
            oi_change_pct = float((oi_change / denom) * 100.0)

            candle = pd.DataFrame([{
                'timestamp': candle_start,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                # REMOVED: 'volume'
                'tick_count': len(ticks),
                'oi': oi_val,
                'oi_change': oi_change,
                'oi_change_pct': oi_change_pct
            }]).set_index('timestamp')

            logger.info(f"[CANDLE] {candle_start.strftime('%H:%M:%S')} OI={oi_val} ΔOI={oi_change} ({oi_change_pct:.2f}%)")
            candle_end = candle_start + timedelta(minutes=interval_min)
            logger.info(f"Candle Created: {candle_start.strftime('%H:%M:%S')}-{candle_end.strftime('%H:%M:%S')} | "
                    f"O:{open_price:.2f} H:{high:.2f} L:{low:.2f} C:{close:.2f} | "
                    f"Ticks:{len(ticks)}")  # CHANGED: Removed volume from log [[20]]

            if self.candle_data.empty:
                self.candle_data = candle
            else:
                self.candle_data = pd.concat([self.candle_data, candle])
                self.candle_data = self.candle_data.tail(self.config.max_candles_stored)
            try:
                self._last_candle_oi = oi_val
                logger.debug(f"[OI] _last_candle_oi set → {self._last_candle_oi}")
            except Exception:
                logger.debug("[OI] _last_candle_oi set skipped")
            self._bucket_closed = True
            if self.on_candle:
                await self.on_candle(candle, self.candle_data)
        except Exception as e:
            logger.error(f"Candle creation error: {e}", exc_info=True)
    
    async def process_messages(self):
        """Main message processing loop with enhanced packet parsing."""
        logger.info("Starting DhanHQ v2 message processing (volume-free)")
        message_count = 0
        error_count = 0
        last_status_log = datetime.now()
        try:
            async for message in self.websocket:
                message_count += 1
                if message_count <= 10:
                    logger.info(f"[process_messages] Received message #{message_count} at {datetime.now(IST).strftime('%H:%M:%S')}")
                try:
                    if isinstance(message, bytes):
                        packet_size = len(message)
                        tick_data = None
                        if packet_size >= 8:
                            response_code = message[0]
                            if response_code == self.TICKER_PACKET and packet_size == 16:
                                tick_data = self._parse_ticker_packet(message)
                            elif response_code == self.QUOTE_PACKET and packet_size >= 50:
                                tick_data = self._parse_quote_packet(message)
                            elif response_code == self.FULL_PACKET and packet_size == 162:
                                tick_data = self._parse_full_packet(message)
                            elif response_code == self.PREV_CLOSE_PACKET and packet_size == 16:
                                tick_data = self._parse_prev_close_packet(message)
                            elif response_code == self.OI_PACKET and packet_size == 12:
                                tick_data = self._parse_oi_packet(message)
                            elif response_code == self.DISCONNECT_PACKET:
                                self._handle_disconnect_packet(message)
                            else:
                                packet_type = self.PACKET_TYPES.get(packet_size, f"unknown_{packet_size}")
                                if packet_size == 8:
                                    tick_data = self._parse_ticker_8(message)
                                elif packet_size == 16:
                                    tick_data = self._parse_quote_16(message)
                                elif packet_size == 32:
                                    tick_data = self._parse_index_full_32(message)
                                elif packet_size == 44:
                                    tick_data = self._parse_equity_full_44(message)
                                elif packet_size == 50 or packet_size == 66:
                                    tick_data = self._parse_quote_packet(message)
                                elif packet_size == 162:
                                    tick_data = self._parse_full_packet(message)
                                elif packet_size == 184:
                                    tick_data = self._parse_market_depth_184(message)
                                else:
                                    self.packet_stats['other'] += 1
                        if tick_data:
                            await self._process_tick(tick_data)
                    else:
                        logger.info(f"[DEBUG] Text message: {message[:200] if message else 'empty'}")
                        await self._handle_text_message(message)
                    if (datetime.now() - last_status_log).total_seconds() > 60:
                        logger.info(f"Status: Messages={message_count}, Errors={error_count}, Ticks={self.tick_count}")
                        logger.info(f"Packet stats: {self.packet_stats}")
                        last_status_log = datetime.now()
                except Exception as e:
                    error_count += 1
                    logger.error(f"Message processing error: {e}")
                    if error_count > 50:
                        logger.critical("Too many errors, attempting reconnection")
                        break
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self.authenticated = False
        except Exception as e:
            logger.error(f"Fatal error in message loop: {e}", exc_info=True)
            if self.on_error:
                await self.on_error(e)
    
    async def _data_watchdog_loop(self):
        """Monitor for data stall; resubscribe once, then reconnect if still stalled."""
        stall_secs = int(getattr(self.config, 'data_stall_seconds', 15))
        retry_secs = int(getattr(self.config, 'data_stall_reconnect_seconds', 30))
        did_resubscribe = False
        logger.info("Starting data-stall watchdog")
        while self.running:
            try:
                await asyncio.sleep(1)
                now = datetime.now(IST)
                if self.last_packet_time is None:
                    if self._last_subscribe_time:
                        since_sub = (now - self._last_subscribe_time).total_seconds()
                        logger.info(f"[Watchdog] {since_sub:.1f}s since subscribe, no packets yet.")
                    if self._last_subscribe_time and (now - self._last_subscribe_time).total_seconds() >= stall_secs and not did_resubscribe:
                        logger.warning(f"No market data for {stall_secs}s after subscribe — re-subscribing")
                        try:
                            await self.subscribe()
                            logger.info(f"[Watchdog] Resubscribe triggered at {now.strftime('%H:%M:%S')}")
                            did_resubscribe = True
                        except Exception as e:
                            logger.error(f"Resubscribe failed: {e}")
                    if self._last_subscribe_time and (now - self._last_subscribe_time).total_seconds() >= retry_secs:
                        logger.warning(f"No market data for {retry_secs}s — reconnecting WebSocket")
                        try:
                            await self.disconnect(stop_running=False)
                        finally:
                            logger.info(f"[Watchdog] Reconnect triggered at {now.strftime('%H:%M:%S')}")
                            break
                else:
                    did_resubscribe = False
            except asyncio.CancelledError:
                logger.debug("Data-stall watchdog cancelled")
                break
            except Exception as e:
                logger.error(f"Data-stall watchdog error: {e}", exc_info=True)
                await asyncio.sleep(2)

    async def run_forever(self):
        """Connect, process, and auto-reconnect until self.running is False."""
        backoff = self.config.reconnect_delay_base
        while self.running:
            try:
                ok = await self.connect()
                if not ok:
                    logger.error("Connect failed, honoring backoff")
                    await asyncio.sleep(min(backoff, 60))
                    backoff = min(backoff * 2, 60)
                    continue
                backoff = self.config.reconnect_delay_base
                await self.process_messages()
                if self.running:
                    logger.warning("Message loop ended; attempting reconnection")
                    await self.disconnect(stop_running=False)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)
            except asyncio.CancelledError:
                logger.info("run_forever cancelled")
                break
            except Exception as e:
                logger.error(f"run_forever error: {e}", exc_info=True)
                if self.running:
                    await asyncio.sleep(min(backoff, 60))
                    backoff = min(backoff * 2, 60)
    
    # Legacy parsers retained (volume-free outputs)
    def _parse_ticker_8(self, data: bytes) -> Optional[Dict]:
        """Parse 8-byte ticker packet (LTP only) - legacy support."""
        if len(data) < 8:
            return None
        try:
            security_id = struct.unpack('<I', data[0:4])[0]
            ltp = struct.unpack('<f', data[4:8])[0]
            if security_id != self.config.nifty_security_id:
                return None
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            logger.debug(f"Legacy Ticker: LTP={ltp:.2f}")
            self._last_best = ltp
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'ticker',
                'ltp': ltp
            }
        except Exception as e:
            logger.error(f"Error parsing ticker packet: {e}")
            return None
    
    def _parse_prev_close_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Previous Close packet (16 bytes)."""
        if len(data) < 16:
            return None
        try:
            response_code = data[0]
            if response_code != self.PREV_CLOSE_PACKET:
                return None
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id == self.config.nifty_security_id:
                prev_close = struct.unpack('<f', data[8:12])[0]
                prev_oi = struct.unpack('<I', data[12:16])[0]
                self.packet_stats['prev_close'] = self.packet_stats.get('prev_close', 0) + 1
                logger.info(f"Previous close: {prev_close:.2f}, Previous OI: {prev_oi}")
                return {
                    "packet_type": "prev_close",
                    "prev_close": prev_close,
                    "prev_oi": prev_oi,
                    "timestamp": datetime.now(IST)
                }
        except Exception as e:
            logger.debug(f"Prev close packet parse error: {e}")
        return None
    
    def _parse_oi_packet(self, data: bytes) -> Optional[Dict]:
        """Parse OI Data packet (12 bytes)."""
        if len(data) < 12:
            return None
        try:
            response_code = data[0]
            if response_code != self.OI_PACKET:
                return None
            message_length = struct.unpack('<H', data[1:3])[0]
            exchange_segment = data[3]
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id == self.config.nifty_security_id:
                oi = struct.unpack('<I', data[8:12])[0]
                self.packet_stats['oi'] = self.packet_stats.get('oi', 0) + 1
                try:
                    self.current_oi = int(oi)
                    logger.debug(f"[OI] Updated from OI packet: {self.current_oi}")
                except Exception:
                    logger.debug("[OI] OI packet update skipped (cast error)")
                return {
                    "packet_type": "oi",
                    "oi": oi,
                    "timestamp": datetime.now(IST)
                }
        except Exception as e:
            logger.debug(f"OI packet parse error: {e}")
        return None
    
    def _handle_disconnect_packet(self, data: bytes):
        """Handle disconnection packet."""
        try:
            if len(data) >= 10:
                disconnect_code = struct.unpack('<H', data[8:10])[0]
                logger.warning(f"Disconnect packet received with code: {disconnect_code}")
        except Exception as e:
            logger.error(f"Disconnect packet handling error: {e}")
    
    def _parse_quote_16(self, data: bytes) -> Optional[Dict]:
        """Parse 16-byte quote packet - legacy support (volume-free)."""
        if len(data) < 16:
            return None
        try:
            packet_type = struct.unpack('<I', data[0:4])[0]
            security_id = struct.unpack('<I', data[4:8])[0]
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
            close = struct.unpack('<f', data[12:16])[0]
            if security_id != self.config.nifty_security_id:
                return None
            if not (self.config.price_sanity_min <= ltp <= self.config.price_sanity_max):
                logger.warning(f"Price sanity check failed: {ltp}")
                return None
            logger.debug(f"Quote16: LTP={ltp:.2f}, Close={close:.2f}")
            self._last_best = ltp
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'quote',
                'ltp': ltp,
                'close': close
            }
        except Exception as e:
            logger.error(f"Error parsing quote packet: {e}")
            return None
    
    def _parse_index_full_32(self, data: bytes) -> Optional[Dict]:
        """Parse 32-byte index full packet - volume-free."""
        if len(data) < 32:
            return None
        try:
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id != self.config.nifty_security_id:
                return None
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
            open_price = struct.unpack('<f', data[12:16])[0]
            high = struct.unpack('<f', data[16:20])[0]
            low = struct.unpack('<f', data[20:24])[0]
            close = struct.unpack('<f', data[24:28])[0]
            logger.debug(f"Index Full: OHLC=[{open_price:.2f},{high:.2f},{low:.2f},{close:.2f}], LTP={ltp:.2f}")
            self._last_best = ltp
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'index_full',
                'ltp': ltp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close
            }
        except Exception as e:
            logger.error(f"Error parsing index full packet: {e}")
            return None
    
    def _parse_equity_full_44(self, data: bytes) -> Optional[Dict]:
        """Parse 44-byte equity/FNO full packet - volume-free."""
        if len(data) < 44:
            return None
        try:
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id != self.config.nifty_security_id:
                return None
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
            open_price = struct.unpack('<f', data[12:16])[0]
            high = struct.unpack('<f', data[16:20])[0]
            low = struct.unpack('<f', data[20:24])[0]
            close = struct.unpack('<f', data[24:28])[0]
            oi = struct.unpack('<I', data[32:36])[0]
            last_traded_time = struct.unpack('<I', data[36:40])[0]
            exchange_time = struct.unpack('<I', data[40:44])[0]
            try:
                self.current_oi = int(oi)
                logger.debug(f"[OI] Updated from EQUITY_FULL_44: {self.current_oi}")
            except Exception:
                logger.debug("[OI] EQUITY_FULL_44 OI update skipped (cast error)")
            logger.info(f"Equity Full: LTP={ltp:.2f}, OI={oi:,}")
            self._last_best = ltp
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'equity_full',
                'ltp': ltp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'oi': oi,
                'ltt': last_traded_time,
                'exchange_time': exchange_time
            }
        except Exception as e:
            logger.error(f"Error parsing equity full packet: {e}")
            return None
    
    def _parse_market_depth_184(self, data: bytes) -> Optional[Dict]:
        """Parse 184-byte market depth packet."""
        if len(data) < 184:
            return None
        try:
            security_id = struct.unpack('<I', data[4:8])[0]
            if security_id != self.config.nifty_security_id:
                return None
            ltp = struct.unpack('<f', data[8:12])[0]
            if not np.isfinite(ltp):
                return None
            logger.debug(f"Market Depth: LTP={ltp:.2f}")
            self._last_best = ltp
            return {
                'timestamp': datetime.now(IST),
                'packet_type': 'market_depth',
                'ltp': ltp
            }
        except Exception as e:
            logger.error(f"Error parsing market depth packet: {e}")
            return None
    
    async def _handle_text_message(self, message: str):
        """Handle text/JSON control messages from server."""
        try:
            data = json.loads(message)
            logger.debug(f"Text message received: {data}")
            if isinstance(data, dict):
                if 'type' in data:
                    msg_type = data['type']
                    if msg_type in ['success', 'subscription_success']:
                        logger.info(f"Server success: {data.get('message', 'Subscription confirmed')}")
                    elif msg_type == 'error':
                        logger.error(f"Server error: {data.get('message', 'Unknown error')}")
                    elif msg_type == 'heartbeat':
                        logger.debug("Heartbeat received")
                    else:
                        logger.debug(f"Server message type: {msg_type}")
                elif 'ResponseCode' in data:
                    code = data['ResponseCode']
                    msg = data.get('ResponseMessage', '')
                    if code == 200:
                        logger.info(f"Subscription successful: {msg}")
                    elif code == 401:
                        logger.critical(f"Authentication failed: {msg}")
                        self.authenticated = False
                    elif code >= 400:
                        logger.error(f"Error {code}: {msg}")
                elif 'market_status' in data:
                    status = data.get('market_status')
                    logger.info(f"Market status: {status}")
                else:
                    logger.debug(f"Unhandled message: {data}")
        except json.JSONDecodeError:
            logger.debug(f"Non-JSON text message received")
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
    
    async def disconnect(self, stop_running: bool = True):
        """Gracefully disconnect from WebSocket."""
        logger.info("Disconnecting from DhanHQ WebSocket")
        if stop_running:
            self.running = False
            logger.debug("Disconnect mode: full shutdown (running=False)")
        else:
            logger.debug("Disconnect mode: internal reconnect (running=True)")
        try:
            if getattr(self, 'boundary_task', None) and not self.boundary_task.done():
                self.boundary_task.cancel()
                await self.boundary_task
        except asyncio.CancelledError:
            logger.debug("Boundary loop task cancelled")
        except Exception as e:
            logger.debug(f"Boundary task cancel failed (ignored): {e}")
        try:
            if getattr(self, 'data_watchdog_task', None) and not self.data_watchdog_task.done():
                self.data_watchdog_task.cancel()
                await self.data_watchdog_task
        except asyncio.CancelledError:
            logger.debug("Data-stall watchdog cancelled")
        except Exception as e:
            logger.debug(f"Data watchdog cancel failed (ignored): {e}")
        if self.websocket:
            try:
                if hasattr(self.websocket, 'open') and hasattr(self.websocket, 'closed'):
                    logger.info(f"[Subscribe] WebSocket state: open={self.websocket.open}, closed={self.websocket.closed}")
                else:
                    logger.info("[Subscribe] WebSocket state: unknown (attributes missing)")
                logger.info(f"Packet statistics: {self.packet_stats}")
                logger.info(f"Total ticks processed: {self.tick_count}")
                await self.websocket.close()
                logger.info("WebSocket disconnected successfully")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        self.authenticated = False

    # ========== NEW: Helper getters for main loop compatibility ==========
    async def get_next_tick(self):
        """Await the next tick from internal queue (tick_data, order_book_snapshot, timestamp)."""
        tick, order_book, ts = await self._tick_queue.get()
        return tick, (order_book or {}), ts

    def get_prices(self, last_n: int = 200):
        """Return recent LTP prices."""
        try:
            return [t.get('ltp') for t in self.tick_buffer[-last_n:] if 'ltp' in t]
        except Exception:
            return []

    def get_order_books(self, last_n: int = 5):
        """Return simplified order book snapshots (best bid/ask from last depth if available)."""
        try:
            if not self._last_depth:
                return []
            # Build a simplified best-level snapshot
            best = self._last_depth[0]
            return [{
                'bid_price': best.get('bid_price', self._last_best),
                'ask_price': best.get('ask_price', self._last_best)
            }]
        except Exception:
            return []

    def get_mid_price(self):
        """Return mid price from last depth or last LTP as fallback."""
        try:
            if self._last_depth:
                best = self._last_depth[0]
                bp = float(best.get('bid_price', self._last_best))
                ap = float(best.get('ask_price', self._last_best))
                return float((bp + ap) / 2.0)
            return float(self._last_best) if self._last_best else 0.0
        except Exception:
            return float(self._last_best) if self._last_best else 0.0

    def get_best_price(self):
        """Return best executable price (fallback to last LTP)."""
        return float(self._last_best) if self._last_best else 0.0

    def get_recent_profit_factor(self):
        """Placeholder for RL modulation; returns neutral value if not implemented."""
        return 1.0

    def get_fill_prob(self):
        """Simple heuristic: always moderate fill probability (placeholder)."""
        return 0.5

    def get_time_waited(self):
        """Placeholder time waited for current order decision."""
        return 0.0

    def get_live_tensor(self):
        """Construct a simple price-normalized tensor from recent prices."""
        try:
            px = np.array(self.get_prices(64), dtype=float)
            if px.size == 0:
                return np.zeros((1, 64, 1), dtype=float)
            px = (px - px.mean()) / max(1e-9, px.std())
            # Right-pad/trim to 64
            if px.size < 64:
                pad = np.zeros(64 - px.size, dtype=float)
                px = np.concatenate([pad, px])
            else:
                px = px[-64:]
            return px.reshape(1, 64, 1)
        except Exception:
            return np.zeros((1, 64, 1), dtype=float)
