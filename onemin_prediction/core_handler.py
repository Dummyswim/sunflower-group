"""
Unified WebSocket handler with integrated microstructure analysis.
Consolidates: enhanced_websocket_handler.py, microstructure_features.py, latency_compensation.py
"""
import asyncio
import struct
import logging
from typing import Dict, Optional, Any, List, Tuple, Callable, Awaitable
import pandas as pd
import numpy as np
import base64
from collections import deque
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


class UnifiedWebSocketHandler:
    """
    All-in-one WebSocket handler with integrated microstructure analysis.
    Eliminates redundancy between handler and extractor classes.
    """
    
    # DhanHQ v2 Packet Types
    PACKET_TYPES = {
        8: "ticker", 16: "quote", 32: "index_full", 44: "equity_full",
        50: "quote_extended", 66: "quote_full", 162: "full_packet",
        184: "market_depth", 492: "market_depth_50"
    }
    
    # Response codes
    TICKER_PACKET = 2
    QUOTE_PACKET = 4
    OI_PACKET = 5
    PREV_CLOSE_PACKET = 6
    MARKET_STATUS_PACKET = 7
    FULL_PACKET = 8
    DISCONNECT_PACKET = 50
    
    def __init__(self, config):
        logger.info("[WS] Initializing Unified WebSocket Handler (volume-free)")

        self.config = config
        self.websocket: Optional[Any] = None
        self.authenticated = False
        self.running = True
        

        # ADD THESE LINES:
        # Core data buffers
        self.tick_buffer = []
        self.current_candle = {'ticks': [], 'start_time': None, 'volume': 0.0}

        # ADD THIS LINE:
        self.candle_data = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Timestamp mode: prefer arrival time to avoid vendor LTT ambiguities
        self.use_arrival_time = bool(getattr(config, "use_arrival_time", True))
        if self.use_arrival_time:
            logger.info("Timestamp mode: ARRIVAL (wall-clock IST)")
        else:
            logger.info("Timestamp mode: VENDOR (normalized LTT)")


        
        # Callbacks
        self.on_tick: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        self.on_candle: Optional[Callable[[pd.DataFrame, pd.DataFrame], Awaitable[None]]] = None
        self.on_error = None
        self.on_preclose: Optional[Callable[[pd.DataFrame, pd.DataFrame], Awaitable[None]]] = None


        # NEW: auth state
        self._auth_event = asyncio.Event()
        self._client_id_masked = None
        self._token_masked = None        

        # INTEGRATED MICROSTRUCTURE (eliminates separate MicrostructureExtractor)
        win = int(getattr(config, 'micro_tick_window', 400))
        self._micro_ticks = deque(maxlen=max(50, win))
        self._quote_lifespans = deque(maxlen=1000)
        self._last_quote_times = {}
        
        # Market state

        self._last_best = None
        self._last_depth = None
        
        # Statistics

        self.last_packet_time = None
        self.tick_count = 0
        
        # Control flags
        self._preclose_fired_for_bucket = None

        self._bucket_closed = False
        self._diag_ticks_left = 50
        
        

        # Rolling PnL / PF history
        self._pnl_hist = deque(maxlen=200)
        self._pf_default = 1.0

                
        
        logger.info(f"Config: SecurityId={config.nifty_security_id}, "
                   f"Interval={config.candle_interval_seconds}s")


        # NEW: decode credentials once for logging/use (masked in logs)
        self._decode_credentials()

    # NEW: util to mask secrets safely for logs
    @staticmethod
    def _mask_secret(s: str, show_start: int = 3, show_end: int = 2) -> str:
        try:
            s = s or ""
            if len(s) <= (show_start + show_end):
                return "*" * len(s) if s else ""
            return f"{s[:show_start]}***{s[-show_end:]}"
        except Exception:
            return "***"

    # NEW: decode and stage credentials from config
    def _decode_credentials(self) -> None:
        try:
            cid_b64 = getattr(self.config, "dhan_client_id_b64", "") or ""
            tok_b64 = getattr(self.config, "dhan_access_token_b64", "") or ""
            client_id = base64.b64decode(cid_b64).decode("utf-8") if cid_b64 else ""
            token = base64.b64decode(tok_b64).decode("utf-8") if tok_b64 else ""
            self._client_id_masked = self._mask_secret(client_id)
            self._token_masked = self._mask_secret(token)
            if client_id and token:
                logger.info(f"Websocket credentials provided: client_id={self._client_id_masked}, token={self._token_masked}")
            else:
                logger.warning("Websocket credentials missing or empty (client_id/token). Authentication will likely fail.")
        except Exception as e:
            logger.error(f"Error decoding credentials: {e}")
            self._client_id_masked = self._token_masked = ""
    
    # NEW: authentication step (stub-safe)
    async def _authenticate(self) -> None:
        """
        Perform authentication/handshake.
        In stub mode: mark authenticated=True if non-empty credentials exist.
        Replace with real auth logic when integrating the live API.
        """
        try:
            # Example real flow (commented, to be implemented when API is wired):
            # await self._perform_ws_handshake()
            # await self._send_auth_message(...)
            # await self._await_auth_ack(...)
            # self.authenticated = True

            # Stub behavior: consider non-empty credentials as 'authenticated'
            tok_b64 = getattr(self.config, "dhan_access_token_b64", "") or ""
            cid_b64 = getattr(self.config, "dhan_client_id_b64", "") or ""
            if tok_b64 and cid_b64:
                self.authenticated = True
                logger.info(f"Websocket authenticated (stub mode) for client_id={self._client_id_masked}")
            else:
                self.authenticated = False
                logger.error("Websocket authentication failed: missing credentials (stub mode)")
        except Exception as e:
            self.authenticated = False
            logger.error(f"Websocket authentication error: {e}", exc_info=True)
        finally:
            # Signal auth result regardless
            try:
                self._auth_event.set()
            except Exception:
                pass



    def _normalize_tick_ts(self, ltt: int) -> datetime:
        """
        Robust normalization of vendor LTT to IST.
        Tries multiple interpretations and chooses the one closest to wall-clock now,
        with a sanity bound. Falls back to arrival time if none are plausible.
        """
        try:
            ltt_int = int(ltt)
        except Exception:
            ltt_int = 0
        
        now_ist = datetime.now(IST)
        candidates = []
        
        # 1) Seconds since IST midnight
        try:
            ist_midnight = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
            t1 = ist_midnight + timedelta(seconds=ltt_int)
            candidates.append(("sod_ist", t1))
        except Exception:
            pass
        
        # 2) Seconds since UTC midnight → IST
        try:
            utc_midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            t2 = (utc_midnight + timedelta(seconds=ltt_int)).astimezone(IST)
            candidates.append(("sod_utc", t2))
        except Exception:
            pass
        
        # 3) Unix epoch seconds → IST
        try:
            t3 = datetime.fromtimestamp(float(ltt_int), tz=timezone.utc).astimezone(IST)
            candidates.append(("epoch_s", t3))
        except Exception:
            pass
        
        # 4) Unix epoch milliseconds → IST
        try:
            t4 = datetime.fromtimestamp(float(ltt_int) / 1000.0, tz=timezone.utc).astimezone(IST)
            candidates.append(("epoch_ms", t4))
        except Exception:
            pass
        
        if not candidates:
            return now_ist
        
        def abs_delta_minutes(ts):
            return abs((ts - now_ist).total_seconds()) / 60.0
        
        # Choose the interpretation closest to now
        mode, ts = min(candidates, key=lambda kv: abs_delta_minutes(kv[1]))
        
        # Sanity bound: require chosen ts to be reasonably close to now (<= 3 minutes).
        # If not, prefer arrival time to avoid drifting buckets.
        if abs_delta_minutes(ts) > 3.0:
            chosen = now_ist
            chosen_mode = "arrival"
        else:
            chosen = ts
            chosen_mode = mode
        
        # One-time diagnostics for first few ticks
        try:
            if not hasattr(self, "_diag_ticks_left"):
                self._diag_ticks_left = 50
            if self._diag_ticks_left > 0:
                self._diag_ticks_left -= 1
                logger.info(
                    f"[TS-NORM] raw_ltt={ltt_int} | now={now_ist.strftime('%H:%M:%S')} | "
                    f"sod_ist={next((t.strftime('%H:%M:%S') for m,t in candidates if m=='sod_ist'), 'na')} | "
                    f"sod_utc={next((t.strftime('%H:%M:%S') for m,t in candidates if m=='sod_utc'), 'na')} | "
                    f"epoch_s={next((t.strftime('%H:%M:%S') for m,t in candidates if m=='epoch_s'), 'na')} | "
                    f"epoch_ms={next((t.strftime('%H:%M:%S') for m,t in candidates if m=='epoch_ms'), 'na')} | "
                    f"chosen={chosen.strftime('%H:%M:%S')} ({chosen_mode})"
                )
        except Exception:
            pass
        
        return chosen




    # ========== INTEGRATED MICROSTRUCTURE ANALYSIS ==========



    def _update_microstructure(self, tick: Dict, order_book: Dict, timestamp: datetime):
        """
        Update microstructure buffers (replaces MicrostructureExtractor.update).
        Integrated to eliminate redundant class.
        """
        try:
            ltp = float(tick.get('ltp', 0.0))
            if ltp > 0:
                # Prevent out-of-order ticks
                if self._micro_ticks and timestamp <= self._micro_ticks[-1][0]:
                    # logger.debug("[MICRO] Out-of-order tick skipped")
                    return
                self._micro_ticks.append((timestamp, ltp))
            
            # Track quote lifespans
            for quote_id, quote in order_book.get('quotes', {}).items():
                if quote_id not in self._last_quote_times:
                    self._last_quote_times[quote_id] = timestamp
                elif quote.get('cancelled'):
                    lifespan = (timestamp - self._last_quote_times[quote_id]).total_seconds()
                    self._quote_lifespans.append(lifespan)
                    del self._last_quote_times[quote_id]
                    
                                
            # Prune stale quote timestamps by age (> 5 minutes) and by size (> 10000)
            try:
                now_ts = timestamp
                
                # Age-based prune
                cutoff_age = timedelta(minutes=5)
                stale_keys = [k for k, ts0 in self._last_quote_times.items() 
                            if (now_ts - ts0) > cutoff_age]
                for k in stale_keys:
                    self._last_quote_times.pop(k, None)
                
                # Size-based prune
                max_qt = 10000
                if len(self._last_quote_times) > max_qt:
                    # Drop oldest keys by age
                    by_age = sorted(self._last_quote_times.items(), key=lambda kv: kv[1])
                    for k, _ in by_age[:len(self._last_quote_times) - max_qt]:
                        self._last_quote_times.pop(k, None)
            except Exception:
                pass
    
                    
        except Exception as e:
            logger.debug(f"[MICRO] Update error: {e}")





    def get_micro_features(self) -> Dict[str, float]:
        """
        Compute ALL microstructure features in one place.
        Consolidates: MicrostructureExtractor methods + handler's get_micro_features.
        """
        out = {
            'imbalance': 0.0, 'slope': 0.0, 'std_dltp': 0.0, 'mean_drift_pct': 0.0,
            'n': 0, 'imbalance_short': 0.0, 'slope_short': 0.0, 'std_dltp_short': 0.0,
            'n_short': 0, 'momentum_delta': 0.0, 'price_range_tightness': 0.0,
            'market_entropy': 0.0, 'quote_lifespan': 0.0
        }
        
        try:
            if not self._micro_ticks:
                return out
            
            # Long window analysis
            win_sec = int(getattr(self.config, 'micro_window_sec_1m', 25))
            min_ticks = int(getattr(self.config, 'micro_min_ticks_1m', 30))
            cutoff = datetime.now(IST) - timedelta(seconds=win_sec)
            recent = [(ts, px) for (ts, px) in list(self._micro_ticks) if ts >= cutoff]
            
            # Short window analysis
            s_sec = int(getattr(self.config, 'micro_short_window_sec_1m', 8))
            s_min = int(getattr(self.config, 'micro_short_min_ticks_1m', 12))
            cutoff_s = datetime.now(IST) - timedelta(seconds=s_sec)
            recent_s = [(ts, px) for (ts, px) in recent if ts >= cutoff_s]
            
            if len(recent) < min_ticks:
                out['n'] = len(recent)
                out['n_short'] = len(recent_s)
                return out
            
            ts, px = zip(*recent)
            n = len(px)
            out['n'] = n
            px_arr = np.asarray(px, dtype=float)
            
            # Price imbalance velocity (replaces trade_imbalance_velocity)
            d = np.diff(px_arr)
            d = d[~np.isclose(d, 0.0)]
            out['std_dltp'] = float(np.nan_to_num(np.std(d))) if d.size else 0.0
            
            ups = int(np.sum(d > 0))
            dns = int(np.sum(d < 0))
            total = ups + dns
            out['imbalance'] = float((ups - dns) / max(1, total)) if total > 0 else 0.0
            out['moves_up'] = ups
            out['moves_dn'] = dns
            out['moves_total'] = total
            
            # Weighted slope
            x = np.arange(n, dtype=float)
            x = (x - x.mean()) / max(1e-9, x.std())
            y = (px_arr - px_arr.mean()) / max(1e-9, px_arr.std())
            w = np.linspace(0.5, 1.0, n)
            out['slope'] = float(np.dot(w * x, y) / max(1e-9, np.dot(w * x, x)))
            
            # Price range tightness (replaces liquidity_absorption_rate)
            rng = float(px_arr.max() - px_arr.min())
            last = float(px_arr[-1])
            base = max(1e-9, abs(last))
            out['price_range_tightness'] = float(1.0 - min(1.0, rng / base))
            
            
            # Market entropy (SAFE: guard empty directions and NaNs)
            directions = np.sign(d)
            # SAFE: guard empty directions and compute entropy without warnings
            if directions.size == 0:
                p_up = 0.0
                p_down = 0.0
            else:
                try:
                    p_up = float(np.mean(directions > 0))
                    p_down = float(np.mean(directions < 0))
                except Exception:
                    p_up, p_down = 0.0, 0.0



            entropy = 0.0
            for p in (p_up, p_down):
                if isinstance(p, (int, float)) and p > 0.0:
                    p_safe = max(1e-12, float(p))
                    entropy_part = -p_safe * np.log2(p_safe)
                    if np.isfinite(entropy_part) and entropy_part >= 0.0:
                        entropy += entropy_part
            if not np.isfinite(entropy) or entropy < 0.0:
                entropy = 0.0
            out['market_entropy'] = float(entropy)

                        
            # Quote lifespan
            out['quote_lifespan'] = float(np.mean(self._quote_lifespans)) if self._quote_lifespans else 0.0
            
            # Micro position
            hi_long = float(np.max(px_arr))
            lo_long = float(np.min(px_arr))
            rng_long = max(1e-9, hi_long - lo_long)
            out['micro_pos'] = float((last - lo_long) / rng_long)
            out['hi_long'] = hi_long
            out['lo_long'] = lo_long
            
            # EMAs
            s = pd.Series(px_arr, dtype='float64')
            out['ema20_1m'] = float(s.ewm(span=min(20, len(s)), adjust=False).mean().iloc[-1])
            out['ema50_1m'] = float(s.ewm(span=min(50, len(s)), adjust=False).mean().iloc[-1])
            out['price_above_ema20'] = bool(last > out['ema20_1m'])
            
            # Last 3s momentum
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
            
            # Mean drift
            mean_px = float(np.mean(px_arr))
            out['mean_drift_pct'] = float(((last - mean_px) / max(1e-9, mean_px)) * 100.0)
            
            # Short window analysis
            out['n_short'] = len(recent_s)
            if len(recent_s) >= s_min:
                _, px_s = zip(*recent_s)
                pxs = np.asarray(px_s, dtype=float)
                d_s = np.diff(pxs)
                d_s = d_s[~np.isclose(d_s, 0.0)]
                out['std_dltp_short'] = float(np.nan_to_num(np.std(d_s))) if d_s.size else 0.0
                
                ups_s = int(np.sum(d_s > 0))
                dns_s = int(np.sum(d_s < 0))
                tot_s = ups_s + dns_s
                out['imbalance_short'] = float((ups_s - dns_s) / max(1, tot_s)) if tot_s > 0 else 0.0
                
                x_s = np.arange(len(pxs), dtype=float)
                x_s = (x_s - x_s.mean()) / max(1e-9, x_s.std())
                y_s = (pxs - pxs.mean()) / max(1e-9, pxs.std())
                w_s = np.linspace(0.6, 1.0, len(pxs))
                out['slope_short'] = float(np.dot(w_s * x_s, y_s) / max(1e-9, np.dot(w_s * x_s, x_s)))
            
            out['momentum_delta'] = float(out['slope_short'] - out['slope'])
            
            # Sanity guards
            if not np.isfinite(out['std_dltp']) or out['std_dltp'] < 0:
                out['std_dltp'] = 0.0
            if out['std_dltp'] > 50.0:
                out['std_dltp'] = 0.0
            if not np.isfinite(out['mean_drift_pct']):
                out['mean_drift_pct'] = 0.0
            if abs(out['mean_drift_pct']) > 2.0:
                out['mean_drift_pct'] = 2.0 if out['mean_drift_pct'] > 0 else -2.0
            
            # logger.info("[MICRO] n=%d|%d | imb=%.2f|%.2f | slp=%.3f|%.3f | drift=%.3f%% | tight=%.2f | entropy=%.2f",
            #            out['n'], out['n_short'], out['imbalance'], out['imbalance_short'],
            #            out['slope'], out['slope_short'], out['mean_drift_pct'],
            #            out['price_range_tightness'], out['market_entropy'])
            
        except Exception as e:
            logger.debug(f"[MICRO] Error: {e}")
        
        return out

    # ========== PACKET PARSING (unchanged, kept for completeness) ==========
    def _parse_ticker_packet(self, data: bytes) -> Optional[Dict]:
        """Parse Ticker packet (16 bytes)."""
        if len(data) < 16:
            return None
        try:
            response_code = data[0]
            if response_code != self.TICKER_PACKET:
                return None
            
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

            # Prefer wall-clock arrival time to avoid vendor LTT ambiguities
            if self.use_arrival_time:
                timestamp = datetime.now(IST)
            else:
                timestamp = self._normalize_tick_ts(ltt)



            self._last_best = ltp
            
            return {
                'timestamp': timestamp,
                'packet_type': 'ticker',
                'ltp': ltp
            }
        except Exception as e:
            logger.error(f"Ticker parse error: {e}")
            return None

    # ========== TICK PROCESSING ==========
    async def _process_tick(self, tick_data: Dict):
        """Process incoming tick with integrated microstructure update."""
        if not tick_data:
            return

        try:
            self.last_packet_time = datetime.now(IST)
            self.tick_count += 1
            self.tick_buffer.append(tick_data)


            # Emit periodic info-level logs so the operator can see ticks arriving
            try:
                if self.tick_count == 1 or (self.tick_count % 50 == 0):
                    ltp_val = float(tick_data.get('ltp', 0.0))
                    ts = tick_data.get("timestamp")
                    if ts is None:
                        ts_str = "n/a"
                    elif hasattr(ts, "isoformat"):
                        ts_str = ts.isoformat()
                    else:
                        ts_str = str(ts)
                    logger.info("[WS] Tick #%d at %s | ltp=%.4f", self.tick_count, ts_str, ltp_val)
            except Exception:
                # Non-fatal instrumentation
                pass




            # INTEGRATED: Update microstructure in one place
            safe_ob = self._last_depth if self._last_depth else {}
            self._update_microstructure(tick_data, safe_ob, tick_data['timestamp'])

            if len(self.tick_buffer) > self.config.max_buffer_size:
                self.tick_buffer.pop(0)

            # Candle bucketing logic
            current_time = tick_data['timestamp']
            interval_min = max(1, self.config.candle_interval_seconds // 60)
            bucket_min = (current_time.minute // interval_min) * interval_min
            candle_start = current_time.replace(minute=bucket_min, second=0, microsecond=0)

            if self.current_candle['start_time'] != candle_start:
                if self.current_candle['start_time'] and self.current_candle['ticks'] and not self._bucket_closed:
                    await self._create_candle(self.current_candle['start_time'], self.current_candle['ticks'])

                self.current_candle = {'start_time': candle_start, 'ticks': [tick_data]}
                self._bucket_closed = False
                self._preclose_fired_for_bucket = None
            else:
                self.current_candle['ticks'].append(tick_data)

            await self._maybe_fire_preclose(current_time)

            if self.on_tick:
                await self.on_tick(tick_data)

        except Exception as e:
            logger.error(f"Tick processing error: {e}", exc_info=True)

    # ========== SIMPLIFIED GETTERS ==========


    def get_prices(self, last_n: int = 200) -> List[float]:
        """Get recent LTP prices."""
        return [t.get('ltp') for t in self.tick_buffer[-last_n:] if 'ltp' in t]

    def get_mid_price(self) -> float:
        """Get mid price from depth or last LTP."""
        try:
            if self._last_depth:
                best = self._last_depth[0]
                bp = float(best.get('bid_price', self._last_best))
                ap = float(best.get('ask_price', self._last_best))
                return (bp + ap) / 2.0
            return float(self._last_best) if self._last_best else 0.0
        except Exception:
            return float(self._last_best) if self._last_best else 0.0

    def get_best_price(self) -> float:
        """Get best executable price."""
        return float(self._last_best) if self._last_best else 0.0

    def get_live_tensor(self) -> np.ndarray:
        """Construct normalized price tensor."""
        try:
            px = np.array(self.get_prices(64), dtype=float)
            if px.size == 0:
                return np.zeros((1, 64, 1), dtype=float)

            px = (px - px.mean()) / max(1e-9, px.std())

            if px.size < 64:
                pad = np.zeros(64 - px.size, dtype=float)
                px = np.concatenate([pad, px])
            else:
                px = px[-64:]

            return px.reshape(1, 64, 1)
        except Exception:
            return np.zeros((1, 64, 1), dtype=float)




    async def _create_candle(self, start_time, ticks: List[Dict[str, Any]]):
        """
        Build an OHLC candle from buffered ticks, append to candle_data, and fire on_candle callback.
        Safe against empty/NaN data and memory bloat.
        """
        try:
            # Extract prices safely
            prices = []
            for t in ticks or []:
                try:
                    p = float(t.get('ltp', 0.0))
                    if np.isfinite(p) and p > 0.0:
                        prices.append(p)
                except Exception:
                    continue

            if not prices:
                # Nothing to build; mark bucket closed to avoid retry storms
                self._bucket_closed = True
                return

            o = float(prices[0])
            h = float(max(prices))
            l = float(min(prices))
            c = float(prices[-1])
            tc = int(len(prices))
            # Volume proxy: use tick_count when real volume is unavailable.
            vol = float(tc)

            # Build single-row DataFrame indexed by start_time
            row = pd.DataFrame([{
                "timestamp": start_time,
                "open": o, "high": h, "low": l, "close": c,
                "volume": vol, "tick_count": tc
            }]).set_index("timestamp")

            # Append to candle_data (single-threaded in asyncio event loop)
            if isinstance(self.candle_data, pd.DataFrame) and not self.candle_data.empty:
                
                self.candle_data = pd.concat([self.candle_data, row], ignore_index=False)

            else:
                self.candle_data = row

            # Enforce memory bound on candle history
            try:
                max_rows = int(getattr(self.config, "max_candles_stored", 2000))
                if max_rows > 0 and len(self.candle_data) > max_rows:
                    self.candle_data = self.candle_data.iloc[-max_rows:]
            except Exception:
                # If config is missing/invalid, keep going with current frame
                pass




            # NEW: Candlestick patterns on last closed candles (1–3)
            try:
                from feature_pipeline import FeaturePipeline  # local import to avoid cycles
                # Use up to 5 recent candles for RVOL baseline; detect on last 1–3
                recent = self.candle_data.tail(5)
                pat = FeaturePipeline.compute_candlestick_patterns(recent)
                if isinstance(pat, dict) and pat:
                    last_idx = self.candle_data.index[-1]
                    # Write pattern features to the last row
                    for k, v in pat.items():
                        try:
                            self.candle_data.loc[last_idx, k] = float(v)
                        except Exception:
                            continue
                    # Recreate row to include pattern columns for callback
                    row = self.candle_data.loc[[last_idx]]
            except Exception as e:
                logger.debug(f"[PAT] Pattern compute skipped: {e}")






            self._bucket_closed = True

            if self.on_candle:
                # Pass a copy of full history to avoid accidental mutation
                full_hist = self.candle_data.copy()
                await self.on_candle(row, full_hist)




        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Candle creation error: {e}", exc_info=True)


    async def _maybe_fire_preclose(self, now_ts):
        """
        Fire a pre-close preview once per candle bucket, in the last preclose_lead_seconds before the candle boundary.
        """
        try:
            on_preclose = self.on_preclose
            if on_preclose is None:
                return
            if not isinstance(self.current_candle, dict):
                return

            start = self.current_candle.get("start_time")
            if start is None:
                return

            # Compute candle close and pre-close trigger time
            try:
                interval_sec = int(getattr(self.config, "candle_interval_seconds", 60))
            except Exception:
                interval_sec = 60
            interval_min = max(1, interval_sec // 60)

            try:
                lead = int(getattr(self.config, "preclose_lead_seconds", 10))
                # Ensure sane lead vs interval
                if lead >= interval_sec:
                    lead = max(1, interval_sec // 2)
            except Exception:
                lead = 10

            close_time = start + timedelta(minutes=interval_min)
            preclose_time = close_time - timedelta(seconds=lead)

            # Fire once per bucket, only after preclose_time
            if now_ts < preclose_time:
                return
            if getattr(self, "_preclose_fired_for_bucket", None) == start:
                return

            ticks = self.current_candle.get("ticks") or []
            prices = []
            for t in ticks:
                try:
                    p = float(t.get("ltp", 0.0))
                    if np.isfinite(p) and p > 0.0:
                        prices.append(p)
                except Exception:
                    continue
            if not prices:
                return

            o, h, l, c = float(prices[0]), float(max(prices)), float(min(prices)), float(prices[-1])
            preview = pd.DataFrame([{
                "timestamp": start,
                "open": o, "high": h, "low": l, "close": c,
                "volume": float(len(prices)), "tick_count": len(prices)
            }]).set_index("timestamp")

            # Mark before awaiting callback to avoid double fire on re-entry
            self._preclose_fired_for_bucket = start

            hist = getattr(self, "candle_data", None)
            full_hist = hist.copy() if isinstance(hist, pd.DataFrame) and not hist.empty else preview.copy()
            await on_preclose(preview, full_hist)

        except asyncio.CancelledError:
            raise
        except Exception:
            # Swallow to avoid noisy logs in tight pre-close windows
            pass




    # ========== ADDED SAFE STUBS AND TASKS ==========
    async def run_forever(self):
        """
        Placeholder run loop. In production, connect to the websocket,
        parse packets, and push ticks via _process_tick. This stub keeps
        the task alive until disconnect() is called.
        """
        logger.info("[WS] run_forever started (stub mode)")


        # NEW: attempt authentication once at start
        await self._authenticate()
                
        try:
            while self.running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("run_forever cancelled")
        except Exception as e:
            logger.error(f"run_forever error: {e}", exc_info=True)
        finally:
            logger.info("[WS] run_forever exiting")


    async def disconnect(self, stop_running: bool = False):
        """Gracefully stop the handler and close resources."""
        try:
            if stop_running:
                self.running = False
            ws = self.websocket
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            logger.info("UnifiedWebSocketHandler disconnected")
        except Exception as e:
            logger.debug(f"disconnect error (ignored): {e}")

    def get_order_books(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """
        Return recent best-depth snapshots for spread calc.
        If depth is not populated, returns an empty list.
        """
        try:
            if not self._last_depth:
                return []
            # If _last_depth is a list of ladders, return a shallow slice
            if isinstance(self._last_depth, list):
                return self._last_depth[-last_n:]
            # If it's a single ladder dict, wrap it
            if isinstance(self._last_depth, dict):
                return [self._last_depth]
        except Exception:
            pass
        return []

    def get_fill_prob(self) -> float:
        """Best-effort estimated fill probability (stub)."""
        try:
            # A basic heuristic could be added here later; default 0.5 is safe
            return 0.5
        except Exception:
            return 0.5

    def get_time_waited(self) -> float:
        """Time waited since last order attempt (stub)."""
        try:
            return 0.0
        except Exception:
            return 0.0


    def record_pnl(self, pnl: float) -> None:
        """Record realized per-candle PnL (points)."""
        try:
            v = float(pnl)
            if np.isfinite(v):
                self._pnl_hist.append(v)
                logger.debug(f"[PF] Recorded PnL={v:.4f} | hist={len(self._pnl_hist)}")
        except Exception:
            pass



    def get_recent_profit_factor(self, window: int = 120) -> float:
        """
        PF = gross_profit / gross_loss over last 'window' scored trades.
        Ignores HOLD/FLAT by design (they shouldn't write PnL).
        """
        try:
            if not self._pnl_hist:
                return self._pf_default
            hist = list(self._pnl_hist)[-max(1, window):]
            gp = sum(x for x in hist if x > 0)
            gl = -sum(x for x in hist if x < 0)
            if gl <= 1e-9:
                pf = 2.0 if gp > 0 else 1.0
            else:
                pf = gp / gl
            logger.debug(f"[PF] window={len(hist)} gp={gp:.4f} gl={gl:.4f} pf={pf:.3f}")
            return float(pf)
        except Exception:
            return self._pf_default
