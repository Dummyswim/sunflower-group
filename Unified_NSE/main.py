"""
Unified Trading System - Main Entry Point
Optimized with proper error handling and type checking
"""
"""
Enhanced Main Entry Point with Persistent Storage and Multi-Timeframe Support
"""
import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union, List
import traceback
import pytz

# Import all modules
from config import UnifiedTradingConfig
from logging_setup import setup_logging, get_logger, log_performance
from data_persistence import DataPersistenceManager  # NEW
from enhanced_websocket_handler import EnhancedWebSocketHandler
from technical_indicators import ConsolidatedTechnicalAnalysis
from consolidated_signal_analyzer import ConsolidatedSignalAnalyzer
from telegram_bot import TelegramBot
from chart_generator import UnifiedChartGenerator
from pattern_detector import CandlestickPatternDetector, ResistanceDetector
from hitrate import HitRateTracker, Candidate # Hit-rate buckets and calibration
from logging_setup import log_span

from contextlib import contextmanager


@contextmanager
def _quiet_logger(name: str, level=logging.WARNING):
    lg = logging.getLogger(name)
    prev = lg.level
    try:
        lg.setLevel(level)
        yield
    finally:
        lg.setLevel(prev)



# Setup logging
setup_logging(
    logfile="logs/unified_trading.log",
    console_level=logging.INFO, # High verbosity during stabilization
    file_level=logging.DEBUG
)

logger = get_logger(__name__)


class EnhancedUnifiedTradingSystem:
    """Enhanced trading system with persistent storage and MTF support."""
    
    def __init__(self):
        """Initialize the enhanced trading system."""
        logger.info("=" * 60)
        logger.info("ENHANCED UNIFIED TRADING SYSTEM INITIALIZATION")
        logger.info("=" * 60)
        
        self.config = None
        self.persistence_manager = None  # NEW
        self.websocket_handler = None
        self.telegram_bot = None
        self.technical_analysis = None
        self.signal_analyzer = None
        self.chart_generator = None
        self.running = True
        self.pattern_detector = None
        self.resistance_detector = None
        self.last_candle_ts_5m = None
        # Track live alerts outcomes (next 3 bars MFE/MAE)
        self.live_alerts = []
        # Pre-close prepared signals cache 
        self._preview_cache: Dict[datetime, Dict[str, Any]] = {}

            

        # Losing streak protection
        self.losing_streak_count = 0
        self.losing_streak_active = False
        self.losing_streak_pause_until = None
        self.recent_trade_outcomes = []  # Track last N outcomes for reset


        self._nm_task = None
        self._nm_lock = asyncio.Lock()  # prevent duplicate :50 predictions

        self._last_nm_predicted: Optional[datetime] = None
        self._last_nm_resolved: Optional[datetime] = None

        
        # self.hit_tracker = HitRateTracker(config=self.config)
        self.hit_tracker = None

        
        logger.info("[HR] Hit-rate tracker pending init (awaiting validated config)")

                
        # Tracking
        self.last_save_time = datetime.now()
        self.stats = {
            'signals_generated': 0,
            'alerts_sent': 0,
            'errors': 0,
            'start_time': datetime.now(),
            'candles_processed_5m': 0,
            'candles_processed_15m': 0
        }
        
        # Multi-timeframe data
        self.candle_aggregator_15m = []  # Aggregate 5-min candles to 15-min

        logger.info("System components initialized")
    
    
    
    

    def _compute_5m_micro_context(self, indicators_5m: dict, session_info: dict) -> float:
        """
        Lightweight 5m context score (0.0-1.0) for 1m micro gating.
        Does NOT inherit 15m alignment — only checks if 5m momentum supports micro.
        """
        try:
            score = 0.5  # neutral baseline
            
            # 1) 5m MACD slope magnitude (momentum strength, not direction)
            macd_closed = float(indicators_5m.get('macd', {}).get('hist_slope_closed', 0.0) or 0.0)
            if abs(macd_closed) >= 0.10:
                score += 0.15  # momentum present
            
            # 2) RSI not at extreme
            rsi5 = float(indicators_5m.get('rsi', {}).get('value', 50.0) or 50.0)
            if 30 < rsi5 < 70:
                score += 0.15  # room to move
            
            # 3) BB position not extreme
            bb_pos = float(indicators_5m.get('bollinger', {}).get('position', 0.5) or 0.5)
            if 0.15 < bb_pos < 0.85:
                score += 0.20  # not at range edge
            
            logger.debug(f"[5m-CONTEXT] score={score:.2f} (macd={abs(macd_closed):.2f}, rsi={rsi5:.1f}, bb={bb_pos:.2f})")
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.debug(f"[5m-CONTEXT] calc error: {e}")
            return 0.5

        

    
    async def initialize(self) -> bool:
        """Initialize all system components with persistent storage."""
        try:
            logger.info("Starting enhanced component initialization...")
            
            # 1. Load configuration
            logger.info("[1/9] Loading configuration...")
            self.config = UnifiedTradingConfig()
            if not self.config.validate():
                logger.error("Configuration validation failed")
                return False

            # Initialize hit-rate tracker with validated config (ensures rotation knobs apply)
            self.hit_tracker = HitRateTracker(config=self.config)
            logger.info("[HR] Hit-rate tracker initialized: base=%s rotate=%s keep_days=%d symlink=%s", self.config.hitrate_base_path, self.config.hitrate_rotate_daily, self.config.hitrate_keep_days, self.config.hitrate_symlink_latest)

            logger.info("✅ Configuration loaded")
            


            
            logger.debug(self.config.get_summary())
            
            # Initialize SetupStats for historical evidence
            from setup_statistics import SetupStats
            self.setup_stats = SetupStats(jsonl_base=self.config.hitrate_base_path)
            self.setup_stats.build()
            logger.info("[EVIDENCE] Setup tables built: recent=%d keys | long=%d keys", len(self.setup_stats.recent), len(self.setup_stats.long))

            
            
            # 2. Initialize Persistent Storage Manager (NEW)
            logger.info("[2/9] Initializing persistent storage...")

            self.persistence_manager = DataPersistenceManager(self.config)
            if not self.persistence_manager.initialize():
                logger.error("Failed to initialize persistent storage")
                return False
            logger.info("✅ Persistent storage initialized")
            
            # 3. Initialize WebSocket handler
            logger.info("[3/9] Initializing WebSocket handler...")
            self.websocket_handler = EnhancedWebSocketHandler(self.config)
            self.websocket_handler.on_candle = self.on_candle_complete
            self.websocket_handler.on_tick = self.on_tick_received
            self.websocket_handler.on_error = self.on_websocket_error
            

                
            # NEW: Set up pre-close handler AFTER websocket is initialized
            
            self.websocket_handler.on_preclose = self.on_preclose_predict 
            logger.info("✓ Pre-close predictive handler wired (on_preclose=on_preclose_predict)")
            

            logger.info("✅ WebSocket handler initialized")


            # CHECK IF THIS SECTION IS ADDED CORRECTLY HERE.
            # Start 1m next-minute loop (evaluation-first)
            if getattr(self.config, 'enable_next_minute_engine', True):
                self._nm_task = asyncio.create_task(self._next_minute_loop())
                logger.info("[NEXT-1m] loop started (predict at :%02d, resolve at :%02d)",
                            int(self.config.next_minute_predict_second),
                            int(self.config.next_minute_resolve_second))
                




            # 4. Initialize Telegram bot
            logger.info("[4/9] Initializing Telegram bot...")
            self.telegram_bot = TelegramBot(
                self.config.telegram_token_b64,
                self.config.telegram_chat_id
            )
            
            # Send startup message
            startup_msg = f"""
<b>🚀 Enhanced Trading System Started</b>

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Mode: Persistent Storage + MTF
5m Data: {len(self.persistence_manager.data_5m)} candles
15m Data: {len(self.persistence_manager.data_15m)} candles
Min Confidence: {self.config.min_confidence}%
Price Action: {self.config.price_action_validation}
MTF Alignment: {self.config.multi_timeframe_alignment}
"""
            self.telegram_bot.send_message(startup_msg)
            logger.info("✅ Telegram bot initialized")
                     
            
            # 5. Initialize technical analysis
            logger.info("[5/9] Initializing technical analysis...")
            self.technical_analysis = ConsolidatedTechnicalAnalysis(self.config)
            logger.info("✅ Technical analysis initialized")
            
            # 6. Initialize pattern detectors (MOVED HERE)
            logger.info("[6/9] Initializing pattern detectors...")
            self.pattern_detector = CandlestickPatternDetector()
            self.resistance_detector = ResistanceDetector()
            logger.info("✅ Pattern detectors initialized")
            
            # 7. Initialize signal analyzer
            logger.info("[7/9] Initializing signal analyzer...")
            self.signal_analyzer = ConsolidatedSignalAnalyzer(
                self.config, 
                self.technical_analysis
            )
            


            try:
                # This assumes self.setup_stats is initialized elsewhere, e.g., self.setup_stats = SetupStats()
                if hasattr(self, 'setup_stats'):
                    self.signal_analyzer.setup_stats = self.setup_stats
                self.signal_analyzer.get_micro_features = getattr(self.websocket_handler, "get_micro_features", None)
                logger.info("[WIRE] setup_stats and micro_features attached to analyzer")
            except Exception:
                pass


            
            # Pass pattern detectors to signal analyzer
            self.signal_analyzer.pattern_detector = self.pattern_detector
        

            self.signal_analyzer.resistance_detector = self.resistance_detector
            logger.info("✅ Signal analyzer initialized")
            
            # 8. Initialize chart generator
            logger.info("[8/9] Initializing chart generator...")
            self.chart_generator = UnifiedChartGenerator(self.config)
            logger.info("✅ Chart generator initialized")
            
            # 9. Run initial analysis if we have data
            logger.info("[9/9] Running initial analysis...")
            await self.run_initial_analysis()
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False
    
    
    
    
    async def _next_minute_loop(self):
        """Fire 1m prediction at :50, resolve previous at :10. Evaluation-first; alerts OFF."""
        try:
            ist = pytz.timezone("Asia/Kolkata")
            while self.running:
                await asyncio.sleep(0.2)
                now = datetime.now(ist)
                sec = now.second
                # Predict at :50
                if sec == int(getattr(self.config, 'next_minute_predict_second', 50)):
                    await self._predict_next_minute(now)
                    # avoid double-fire in same second
                    await asyncio.sleep(1.0)
                # Resolve at :10 → previous minute window
                if sec == int(getattr(self.config, 'next_minute_resolve_second', 10)):
                    await self._resolve_last_minute(now)
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.debug("[NEXT-1m] loop cancelled")
        except Exception as e:
            logger.error(f"[NEXT-1m] loop error: {e}", exc_info=True)





    async def _predict_next_minute(self, now_ist: datetime):
        """
        Predict next 1m bar at :predict_second with high-visibility, de-duplicated logging.
        Flow: ENTRY → FRAMES → INDICATORS → CONTEXT → NEXT-1m → SETUPS → MICRO SAVE → CANDIDATE SAVE
        """
        try:
            async with self._nm_lock:
                predict_sec = int(getattr(self.config, 'next_minute_predict_second', 57))
                resolve_sec = int(getattr(self.config, 'next_minute_resolve_second', 10))

                # 1) ENTRY and duplicate-guard
                next_start = now_ist.replace(second=0, microsecond=0) + timedelta(minutes=1)
                logger.info("[NEXT-1m][ENTRY] now=%s | target_next_start=%s | predict_sec=:%02d | resolve_sec=:%02d",
                            now_ist.strftime('%H:%M:%S'), next_start.strftime('%H:%M'), predict_sec, resolve_sec)
                if self._last_nm_predicted == next_start:
                    logger.debug("[NEXT-1m][ENTRY] Duplicate guard: already predicted %s", next_start.strftime('%H:%M'))
                    return
                self._last_nm_predicted = next_start

                # 2) Reset per-minute noise baseline for 1m analyzer
                try:
                    if self.signal_analyzer:
                        self.signal_analyzer._nm_std_ref = None
                        logger.debug("[NEXT-1m][INIT] Analyzer 1m std baseline reset")
                except Exception as e:
                    logger.debug("[NEXT-1m][INIT] Baseline reset skipped: %s", e)

                # 3) MICRO snapshot (always try; used even if setups fail later)
                try:
                    micro = self.websocket_handler.get_micro_features() if self.websocket_handler else {}
                except Exception as e:
                    logger.debug("[NEXT-1m][MICRO] snapshot failed: %s", e)
                    micro = {}

                # 4) FRAMES: persisted 5m and 15m, plus ws-tail fallback for setup derivation
                df_5m = pd.DataFrame()
                df_15m = pd.DataFrame()
                try:
                    df_5m = self.persistence_manager.get_data("5m", 100)
                    df_15m = self.persistence_manager.get_data("15m", 60)
                except Exception as e:
                    logger.error("[NEXT-1m][FRAME] persist get_data error: %s", e)

                # ws-tail (fresh) frame for setups
                df_5m_ws = None
                try:
                    if (self.websocket_handler and isinstance(self.websocket_handler.candle_data, pd.DataFrame)
                        and not self.websocket_handler.candle_data.empty):
                        df_5m_ws = self.websocket_handler.candle_data.tail(100).copy()
                except Exception as e:
                    logger.debug("[NEXT-1m][FRAME] ws-tail fetch skipped: %s", e)

                # Choose frame for setup detectors
                df_5m_for_setups = df_5m
                persist_ref = df_5m.index[-1].strftime('%H:%M') if not df_5m.empty else "no persisted 5m"
                ws_ref = (df_5m_ws.index[-1].strftime('%H:%M') if (df_5m_ws is not None and not df_5m_ws.empty)
                        else "no websocket 5m")

                try:
                    if (df_5m_ws is not None and not df_5m_ws.empty and not df_5m.empty
                            and df_5m_ws.index[-1] > df_5m.index[-1]):
                        df_5m_for_setups = df_5m_ws
                        logger.info("[NEXT-1m][FRAME] ws ahead of persist → setups will use ws-frame (persist=%s, ws=%s)", persist_ref, ws_ref)
                    else:
                        logger.info("[NEXT-1m][FRAME] setups will use persist-frame (persist=%s, ws=%s)", persist_ref, ws_ref)
                except Exception as e:
                    logger.debug("[NEXT-1m][FRAME] selection fallback to persist: %s", e)

                # 5) INDICATORS (persisted 5m and 15m)
                indicators_5m = {}
                indicators_15m = None
                if df_5m.empty:
                    logger.info("[NEXT-1m][IND] No 5m data available; aborting this :%02d cycle", predict_sec)
                    return
                try:
                    indicators_5m = await self.technical_analysis.calculate_all_indicators(df_5m)
                    if not df_15m.empty:
                        indicators_15m = await self.calculate_15m_indicators(df_15m)
                    logger.info("[NEXT-1m][IND] Indicators computed: 5m=%d blocks | 15m=%s", len(indicators_5m), "yes" if indicators_15m else "no")
                except Exception as e:
                    logger.error("[NEXT-1m][IND] Indicator calc error: %s", e)
                    return

                # 6) CONTEXT (session characteristics and MTF score for visibility only)
                session_info = self.signal_analyzer.detect_session_characteristics(df_5m)
                try:
                    self.signal_analyzer.current_df = df_5m  # scorer ref
                except Exception:
                    pass

                mtf_score = 0.0
                try:
                    checker = (self.signal_analyzer.mtf_analyzer.check_timeframe_alignment_quiet
                            if getattr(self.config, 'quiet_mtf_in_1m', True)
                            else self.signal_analyzer.mtf_analyzer.check_timeframe_alignment)
                    mtf_aligned, mtf_score, _desc = (
                        checker({'composite_signal': 'NEUTRAL'}, indicators_15m, df_15m, session_info)
                        if indicators_15m is not None and not df_15m.empty else (True, 0.0, "no-15m")
                    )
                except Exception:
                    mtf_score = 0.0

                # Use only 5m context score for next-1m gating
                micro_context_score = self._compute_5m_micro_context(indicators_5m, session_info)
                logger.info("[NEXT-1m][CTX] session=%s | micro_ctx=%.2f | mtf_score(15m view)=%.2f",
                            session_info.get('session', 'unknown'), micro_context_score, mtf_score)

                # 7) NEXT-1m forecast/classifier (single call; sanitized in analyzer)
                nm = self.signal_analyzer.analyze_next_minute(
                    indicators_5m, indicators_15m, micro, micro_context_score,
                    "open" if now_ist.hour*60+now_ist.minute <= 10*60+15 else "mid" if now_ist.hour*60+now_ist.minute < 14*60+30 else "close"
                )


                # Store ML example for later learning (at resolve)
                try:
                    if nm.get('nm_features') is not None and nm.get('model_p') is not None:
                        self.signal_analyzer.remember_nm_example(next_start, np.array(nm['nm_features'], dtype=float), float(nm['model_p']))
                except Exception as e:
                    logger.debug("[NM-ML] remember example skipped: %s", e)


                # 8) FORECAST (display-only) summary
                side_cls = str(nm.get('composite_signal', 'NEUTRAL')).upper()
                forecast_dir = str(nm.get('forecast_color', side_cls)).upper()
                forecast_prob = float(nm.get('forecast_prob', nm.get('confidence', 30.0)))
                logger.info("[NEXT-1m][FORECAST] t+1m=%s | dir=%s | prob=%.1f%% | classifier_side=%s",
                            next_start.strftime('%H:%M'), forecast_dir, forecast_prob, side_cls)

                # 9) SETUPS: derive directly on freshest 5m frame; provide meaningful messages
                pivot_dir = None
                pivot_name = None
                imb_dir = None
                imb_why = None
                try:
                    self.signal_analyzer.current_df = df_5m_for_setups
                except Exception:
                    pass

                # Detectors (safe)
                ps_det = {}
                imb_det = {}
                try:
                    ps_det = self.signal_analyzer.detect_pivot_swipe(df_5m_for_setups, indicators_5m, self.config) or {}
                except Exception as e:
                    logger.debug("[NEXT-1m][SETUPS] pivot_swipe detect skipped: %s", e)
                try:
                    imb_det = self.signal_analyzer.detect_imbalance_structure(df_5m_for_setups, indicators_5m, self.config) or {}
                except Exception as e:
                    logger.debug("[NEXT-1m][SETUPS] imbalance detect skipped: %s", e)

                pivot_dir = ps_det.get('direction')          # LONG/SHORT or None
                pivot_name = ps_det.get('level_name')        # PDH/PDL/SWING_H/SWING_L or None
                imb_dir = imb_det.get('direction')           # LONG/SHORT or None
                imb_why = imb_det.get('why')                 # reason if not detected

                # 1m micro-direction hint when 5m imbalance setup is absent
                micro_hint = None
                try:
                    imbL = float(micro.get('imbalance', 0.0) or 0.0)
                    imbS = float(micro.get('imbalance_short', 0.0) or 0.0)
                    nS = int(micro.get('n_short', 0) or 0)
                    wS = 0.60 if nS >= int(getattr(self.config, 'micro_short_min_ticks_1m', 12)) else 0.0
                    wL = 0.40
                    blend = (wS*imbS + wL*imbL) / max(1e-9, wS + wL)
                    if abs(blend) >= 0.15:
                        micro_hint = 'LONG' if blend > 0 else 'SHORT'
                except Exception:
                    micro_hint = None

                # Build friendly "meaningful messages"
                swipe_msg = (f"{pivot_dir} @{pivot_name}" if (pivot_dir and pivot_name)
                            else "no swipe on last closed 5m bar (no PDH/PDL/SWING reclaim)")
                if imb_dir:
                    imb_msg = f"{imb_dir}"
                else:
                    imb_msg = ("no 5m imbalance setup on last closed bar"
                            + (f" (why={imb_why})" if imb_why else ""))

                # Include both frames for transparency
                logger.info("[NEXT-1m][SETUPS] 5m_ref(persist)=%s | 5m_ref(ws)=%s | swipe=%s | imbalance_5m=%s%s",
                            persist_ref, ws_ref, swipe_msg,
                            imb_msg,
                            (f" | micro_hint={micro_hint}" if (not imb_dir and micro_hint) else ""))

                # 10) Build and save Candidate for hit-rate (forecast-only evaluation)
                cand = Candidate(
                    next_bar_time=next_start,
                    direction="NEUTRAL",
                    actionable=False,
                    rejection_reason="1m_eval",
                    mtf_score=float(mtf_score),
                    breadth=int(0),
                    weighted_score=0.0,
                    macd_hist_slope=float(indicators_5m.get('macd', {}).get('hist_slope_closed', 0.0) or 0.0),
                    rsi_value=float(indicators_5m.get('rsi', {}).get('value', 50.0) or 50.0),
                    rsi_cross_up=False,
                    rsi_cross_down=False,
                    pattern_used=False,
                    sr_room=str(getattr(self.signal_analyzer.mtf_analyzer, '_last_sr_room', 'UNKNOWN')),
                    regime=str(session_info.get('session', 'UNKNOWN')),
                    confidence=float(forecast_prob) / 100.0,
                    oi_signal=(indicators_5m.get('oi', {}) or {}).get('signal', 'neutral'),
                    oi_change_pct=float((indicators_5m.get('oi', {}) or {}).get('oi_change_pct', 0.0) or 0.0),
                    saved_at=datetime.now(),
                    horizon="1m",
                    liberal_direction=("BUY" if "BUY" in side_cls else ("SELL" if "SELL" in side_cls else "NEUTRAL"))
                )

                # Attach setup tags (with reasons and hint)
                try:
                    cand.pivot_swipe = pivot_dir
                    cand.pivot_level = pivot_name
                    cand.imbalance = imb_dir               # 5m setup (LONG/SHORT or None)
                    cand.imbalance_why = imb_why           # reason when None
                    cand.imbalance_micro_hint = micro_hint # 1m micro hint when 5m setup missing
                except Exception as e:
                    logger.debug("[NEXT-1m][SETUPS] attach skipped: %s", e)

                # Attach :57 micro snapshot (both windows) for calibration
                try:
                    cand.nm_micro = {
                        'imbalance': float(micro.get('imbalance', 0.0) or 0.0),
                        'imbalance_short': float(micro.get('imbalance_short', 0.0) or 0.0),
                        'slope': float(micro.get('slope', 0.0) or 0.0),
                        'slope_short': float(micro.get('slope_short', 0.0) or 0.0),
                        'momentum_delta': float(micro.get('momentum_delta', 0.0) or 0.0),
                        'std_dltp': float(micro.get('std_dltp', 0.0) or 0.0),
                        'std_dltp_short': float(micro.get('std_dltp_short', 0.0) or 0.0),
                        'drift': float(micro.get('vwap_drift_pct', 0.0) or 0.0),
                        'n': int(micro.get('n', 0) or 0),
                        'n_short': int(micro.get('n_short', 0) or 0)
                    }
                    logger.info("[NEXT-1m][MICRO] saved: n=%d|%d imbL=%.2f imbS=%.2f slpL=%.3f slpS=%.3f momΔ=%.3f stdL=%.5f stdS=%.5f drift=%.3f%%",
                                cand.nm_micro['n'], cand.nm_micro['n_short'],
                                cand.nm_micro['imbalance'], cand.nm_micro['imbalance_short'],
                                cand.nm_micro['slope'], cand.nm_micro['slope_short'],
                                cand.nm_micro['momentum_delta'], cand.nm_micro['std_dltp'],
                                cand.nm_micro['std_dltp_short'], cand.nm_micro['drift'])
                except Exception as e:
                    logger.debug("[NEXT-1m][MICRO] attach skipped: %s", e)

                # Finally save candidate to hit-rate
                self.hit_tracker.save_candidate(cand)
                logger.info("[NEXT-1m][SAVE] Candidate saved for t+1m=%s (forecast=%s @ %.1f%%)",
                            next_start.strftime('%H:%M'), forecast_dir, forecast_prob)

            
                # Store ML example for later learning (at resolve)
                try:
                    if nm.get('nm_features') is not None and nm.get('model_p') is not None:
                        self.signal_analyzer.remember_nm_example(next_start, np.array(nm['nm_features'], dtype=float), float(nm['model_p']))
                except Exception as e:
                    logger.debug("[NM-ML] remember example skipped: %s", e)


        except Exception as e:
            logger.error(f"[NEXT-1m] predict error: {e}", exc_info=True)




    async def _resolve_last_minute(self, now_ist: datetime):
        """
        Resolve the previous 1m bar using tick data in [prev_start, prev_start+1m).
        Flow: ENTRY → WINDOW → TICKS → BAR → SAVE
        High-visibility logs; idempotent and race-safe under _nm_lock.
        """
        try:
            async with self._nm_lock:
                # 1) ENTRY and window
                window_end = now_ist.replace(second=0, microsecond=0)  # current minute start
                prev_start = window_end - timedelta(minutes=1)  # previous minute start
                logger.info("[NEXT-1m][RESOLVE][ENTRY] now=%s | window=%s→%s",
                            now_ist.strftime('%H:%M:%S'),
                            prev_start.strftime('%H:%M:%S'),
                            window_end.strftime('%H:%M:%S'))

                # Idempotent guard
                if self._last_nm_resolved == prev_start:
                    logger.debug("[NEXT-1m][RESOLVE][ENTRY] Duplicate guard: already resolved %s",
                                prev_start.strftime('%H:%M'))
                    return

                # 2) Safety: no handler
                if not self.websocket_handler:
                    logger.info("[NEXT-1m][RESOLVE] No WebSocket handler available; skipping resolution")
                    return

                # 3) Pull ticks in the exact minute window [prev_start, window_end)
                try:
                    ticks = self.websocket_handler.get_ticks_between(prev_start, window_end) or []
                except Exception as e:
                    logger.debug("[NEXT-1m][RESOLVE] get_ticks_between failed: %s", e)
                    ticks = []

                if not ticks:
                    logger.info("[NEXT-1m][RESOLVE][MISS] 0 ticks in window %s→%s; will retry next :%02d",
                                prev_start.strftime('%H:%M:%S'),
                                window_end.strftime('%H:%M:%S'),
                                int(getattr(self.config, 'next_minute_resolve_second', 10)))
                    return

                # 4) Sanitize and sort by timestamp; keep only finite prices > 0
                try:
                    valid = [t for t in ticks
                             if t.get('timestamp') is not None
                             and isinstance(t.get('ltp'), (int, float))
                             and float(t.get('ltp')) > 0]
                    ticks_sorted = sorted(valid, key=lambda x: x['timestamp'])
                except Exception:
                    ticks_sorted = []

                if not ticks_sorted:
                    logger.info("[NEXT-1m][RESOLVE][MISS] No valid price ticks in window; deferring")
                    return

                first_ts = ticks_sorted[0]['timestamp']
                last_ts = ticks_sorted[-1]['timestamp']
                open_px = float(ticks_sorted[0]['ltp'])
                close_px = float(ticks_sorted[-1]['ltp'])

                # 5) Diagnostics on coverage inside the window
                try:
                    gap_pre = (first_ts - prev_start).total_seconds()
                    gap_post = (window_end - last_ts).total_seconds()
                except Exception:
                    gap_pre = gap_post = 0.0

                logger.info("[NEXT-1m][RESOLVE][TICKS] count=%d | first=%s (+%.1fs) | last=%s (-%.1fs)",
                            len(ticks_sorted),
                            first_ts.strftime('%H:%M:%S'),
                            gap_pre,
                            last_ts.strftime('%H:%M:%S'),
                            gap_post)

                # 6) Compute bar outcome
                move = close_px - open_px
                direction = "UP" if move > 0 else ("DOWN" if move < 0 else "FLAT")
                logger.info("[NEXT-1m][RESOLVE][BAR] open=%.2f close=%.2f Δ=%.2f (%s)",
                            open_px, close_px, move, direction)


                # Online learning for 1m ML model (label = 1 if UP else 0)
                try:
                    if hasattr(self.signal_analyzer, '_nm_pending'):
                        ex = self.signal_analyzer._nm_pending.pop(prev_start, None)
                        if ex and self.signal_analyzer.nm_model is not None:
                            y = 1 if move > 0 else 0
                            self.signal_analyzer.nm_model.update(ex['x'], y)
                            logger.info("[NM-ML] learned @%s | y=%d | p_ml(prev)=%.3f", prev_start.strftime('%H:%M'), y, float(ex.get('p_ml', 0.0)))
                except Exception as e:
                    logger.debug("[NM-ML] learn skipped: %s", e)
                


                # 7) Persist result into hit-rate and mark resolved
                try:
                    self.hit_tracker.resolve_bar(prev_start, open_px, close_px, logger, horizon="1m")
                except Exception as e:
                    logger.error("[NEXT-1m][RESOLVE] hit_tracker.resolve_bar error: %s", e, exc_info=True)

                self._last_nm_resolved = prev_start
                logger.info("[NEXT-1m][RESOLVE][DONE] resolved_minute=%s", prev_start.strftime('%H:%M'))

        except Exception as e:
            logger.error("[NEXT-1m][RESOLVE] error: %s", e, exc_info=True)








    async def send_predictive_alert(self, signal: Dict, next_candle_start: datetime, df: pd.DataFrame) -> None:
        """Send alert PREDICTING the next candle."""
        try:
            import pytz
            ist = pytz.timezone("Asia/Kolkata")
            now_ist = datetime.now(ist)
            
            # Format next candle time window
            next_candle_end = next_candle_start + timedelta(minutes=5)
            
            # Get current price
            current_price = float(df['close'].iloc[-1]) if not df.empty else 0
            
            # Determine signal strength
            signal_type = signal.get('composite_signal', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            weighted_score = signal.get('weighted_score', 0)
            
            if 'STRONG_BUY' in signal_type:
                emoji = "🟢🟢🟢"
                action = "STRONG BUY"
                prediction = "Expecting +20-30 points"
            elif 'BUY' in signal_type:
                emoji = "🟢"
                action = "BUY"
                prediction = "Expecting +10-20 points"
            elif 'STRONG_SELL' in signal_type:
                emoji = "🔴🔴🔴"
                action = "STRONG SELL"
                prediction = "Expecting -20-30 points"
            elif 'SELL' in signal_type:
                emoji = "🔴"
                action = "SELL"
                prediction = "Expecting -10-20 points"
            else:
                logger.info("Neutral prediction - not sending alert")
                return
            

            size_mult = 0.0
            try:
                size_mult = float(signal.get('entry_exit', {}).get('size_multiplier', signal.get('size_multiplier', 0.0)))
            except Exception:
                size_mult = 0.0

            
            message = f"""
    {emoji} <b>NEXT CANDLE PREDICTION</b> {emoji}

    <b>⏰ PREDICTION FOR:</b>
    {next_candle_start.strftime('%H:%M')}-{next_candle_end.strftime('%H:%M')} IST

    <b>🎯 FORECAST:</b>
    • Signal: {action}
    • {prediction}
    • Confidence: {confidence:.1f}%
    • Score: {weighted_score:.3f}

    <b>📊 CURRENT STATUS:</b>
    • Price Now: ₹{current_price:,.2f}
    • Analysis Time: {now_ist.strftime('%H:%M:%S')}
    • Based on: Incomplete {next_candle_start.strftime('%H:%M')} candle

    <b>💡 ENTRY STRATEGY:</b>
    • Entry: At {next_candle_start.strftime('%H:%M:00')} candle open
    • Stop Loss: {signal.get('stop_loss', 0):.2f}
    • Target: {signal.get('take_profit', 0):.2f}
    • Risk/Reward: {signal.get('risk_reward', 0):.2f} 
    • Duration: ~{signal.get('duration_prediction', {}).get('estimated_minutes', 10)} min (e.g., {next_candle_start.strftime('%H:%M')}-{(next_candle_start + timedelta(minutes=signal.get('duration_prediction', {}).get('estimated_minutes', 10))).strftime('%H:%M')})
    • Size Multiplier: {size_mult:.2f}x


    <b>📈 INDICATORS:</b>
    • Active: {signal.get('active_indicators', 0)}/6
    • MTF Aligned: {signal.get('mtf_analysis', {}).get('aligned', False)}

    <i>⚠️ This is a PREDICTION for the upcoming candle</i>
    <i>📍 Entry at candle open, not current price</i>
    """
            
            # Send alert
            success = self.telegram_bot.send_message(message)

            if success:
                logger.info(f"[ALERT] size_multiplier (pre-close): {size_mult:.2f}")

                logger.info(f"✅ Predictive alert sent → NEXT: {next_candle_start.strftime('%H:%M')}-{next_candle_end.strftime('%H:%M')} IST (entry at open) | {signal_type} @ {confidence:.1f}%")

            else:
                logger.error("Failed to send predictive alert")
            
        except Exception as e:
            logger.error(f"Predictive alert error: {e}", exc_info=True)



    async def on_preclose_predict(self, preview_candle: pd.DataFrame, all_candles: pd.DataFrame) -> None: 
        """Prepare next-candle forecast before the current bar closes.""" 
        try: 
            liberal_direction = "NEUTRAL"
            if preview_candle is None or preview_candle.empty: 
                return 
            preview_start = preview_candle.index[0] 
            preview_close = preview_start + timedelta(seconds=self.config.candle_interval_seconds)
            logger.info("=" * 60)
        
            logger.info(f"🔔 PRE-CLOSE ANALYSIS: start={preview_start.strftime('%H:%M:%S')} "
                        f"close={preview_close.strftime('%H:%M:%S')}")
            logger.info("=" * 60)
        
            # Build working 5m = stored + preview (do not persist)
            df_5m = self.persistence_manager.get_data("5m", 100)
            try:
                df_5m = pd.concat([df_5m, preview_candle]).sort_index()
                df_5m = df_5m[~df_5m.index.duplicated(keep='last')]
            except Exception:
                pass

            # Build 15m from stored 5m (resample) up to preview close
            df_15m = pd.DataFrame()
            try:
                if not self.persistence_manager.data_5m.empty:         

                    df_15m_resampled = (
                        self.persistence_manager.data_5m.tail(800)
                        .resample('15min', label='left', closed='left', origin='start_day', offset='15min')
                        .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
                        .dropna()
                    )
          
                    
                    df_15m_resampled.index = df_15m_resampled.index - pd.Timedelta(minutes=15)
                    df_15m = df_15m_resampled[df_15m_resampled.index <= preview_close].tail(50).copy()
            except Exception as e:
                logger.debug(f"15m resample (preview) failed, fallback to stored 15m: {e}")
                df_15m = self.persistence_manager.get_data("15m", 50)

            # Indicators and analysis
            indicators_5m = await self.technical_analysis.calculate_all_indicators(df_5m)
            indicators_15m = await self.calculate_15m_indicators(df_15m) if not df_15m.empty else None

            try:
                if not df_15m.empty:
                    last_htf = df_15m.index[-1]
                    logger.info("Sentinel: Using last CLOSED 15m bar only → %s", last_htf.strftime('%H:%M:%S'))

            except Exception:
                pass



            signal = await self.signal_analyzer.analyze_and_generate_signal(
                indicators_5m, df_5m, indicators_15m, df_15m
            )

            # Always build a calibration candidate (even when action is rejected or signal is None)
            session_info = self.signal_analyzer.detect_session_characteristics(df_5m)
            logger.debug(f"[Pre-close] Session snapshot: {session_info}")
            
            logger.info(f"[Pre-close] Session snapshot: {session_info}")


            contrib = {}
            w_score = 0.0
            direction = "NEUTRAL"
            conf = 0.0
            breadth = 0
            macd_slope = 0.0
            rsi_val = 50.0
            rsi_up = False
            rsi_dn = False
            pattern_used = False
            sr_room = "UNKNOWN"
            mtf_score = 0.0
            rej = None

            try:
                if signal:
                    contrib = signal.get('contributions', {})
                    w_score = float(signal.get('weighted_score', 0.0))
                    direction = str(signal.get('composite_signal', 'NEUTRAL'))
                    conf = float(signal.get('confidence', 0.0))
                    breadth = int(signal.get('active_indicators', 0))
                    mtf_score = float(signal.get('mtf_score', 0.0))
                    sr_room = str(signal.get('mtf_analysis', {}).get('sr_room', 'UNKNOWN'))
                    sr_room = sr_room or getattr(self.signal_analyzer.mtf_analyzer, '_last_sr_room', 'UNKNOWN')


                    rej = signal.get('rejection_reason')
                    
                    if isinstance(rej, str) and rej.upper() in {'PA','MTF','VALIDATION','CONFIDENCE','RR'}:
                        mapper = {'PA': 'pa_veto', 'MTF': 'mtf_not_aligned', 'VALIDATION': 'validation', 'CONFIDENCE': 'confidence_floor', 'RR': 'rr_floor', 'MOMENTUM': 'momentum_exhaustion'}
                        rej = mapper[rej.upper()]
                    if not rej and getattr(self.signal_analyzer, "_last_reject", None):
                        stage = str(self.signal_analyzer._last_reject.get('stage', '')).upper()
                        mapper = {'PA': 'pa_veto', 'MTF': 'mtf_not_aligned', 'VALIDATION': 'validation', 'CONFIDENCE': 'confidence_floor', 'RR': 'rr_floor', 'MOMENTUM': 'momentum_exhaustion'}
                        rej = mapper.get(stage, stage.lower() or None)



                                        
                else:
                    # Compute a lightweight raw scorer for calibration (no validations)
                    try:
                        self.signal_analyzer.current_df = df_5m
                    except Exception:
                        pass
                    raw = self.signal_analyzer._calculate_weighted_signal(indicators_5m)

                    
                    contrib = raw.get('contributions', {}) 
                    w_score = float(raw.get('weighted_score', 0.0)) 
                    direction = str(raw.get('composite_signal', 'NEUTRAL')) 
                    breadth = int(raw.get('active_indicators', 0)) # FIX: carry actual active indicator count # Compute MTF score … 
                    try: 
                    
                        
                        mtf_aligned, mtf_score, _ = self.signal_analyzer.mtf_analyzer.check_timeframe_alignment_quiet( 
                            raw, indicators_15m, df_15m, session_info 
                        ) if indicators_15m and df_15m is not None else (True, 0.0, "")
                        
                        
                        
                    except Exception: 
                        mtf_score = 0.0










                    sr_room = getattr(self.signal_analyzer.mtf_analyzer, '_last_sr_room', 'UNKNOWN')


                    # If analyzer rejected earlier, copy normalized rejection reason for traceability
                    try:
                        if getattr(self.signal_analyzer, "_last_reject", None):
                            stage = str(self.signal_analyzer._last_reject.get('stage', '')).upper()
                            mapper = {'PA': 'pa_veto', 'MTF': 'mtf_not_aligned', 'VALIDATION': 'validation', 'CONFIDENCE': 'confidence_floor', 'RR': 'rr_floor', 'MOMENTUM': 'momentum_exhaustion'}
                            rej = mapper.get(stage, stage.lower() or None)
                    except Exception:
                        pass



                macd_slope = float(contrib.get('macd', {}).get('hist_slope', 0.0))
                rsi_val = float(contrib.get('rsi', {}).get('rsi_value', 50.0))
                rsi_up = bool(contrib.get('rsi', {}).get('rsi_cross_up', False))
                rsi_dn = bool(contrib.get('rsi', {}).get('rsi_cross_down', False))
                pattern_used = bool(signal and signal.get('pattern_boost_applied', False))
            except Exception:
                pass


            try:
                rsi_fallback = bool(contrib.get('rsi', {}).get('fallback', False))
                logger.info(f"[CAL] rsi_fallback={rsi_fallback}")
            except Exception:
                pass


            try:
                oi_ctx = contrib.get('oi', {}) or {}
                cand_oi_signal = str(oi_ctx.get('signal', 'neutral'))
                cand_oi_chg = float(oi_ctx.get('oi_change_pct', 0.0))
            except Exception:
                cand_oi_signal = None
                cand_oi_chg = 0.0
            logger.info(f"[Pre-close] OI ctx: {cand_oi_signal} ΔOI%={cand_oi_chg:.2f}")



            eligible = False
            try:
                if signal:
                    # Evaluate alert-eligibility without actually sending
                    eligible = await self._should_send_alert(signal)
                    logger.info("[PRECLOSE] Eligibility (no-send) → %s", eligible)

            except Exception as e:
                logger.debug(f"[PRECLOSE] Eligibility check skipped: {e}")


            # Liberal shadow classification (pre-gate; eval-only)
            try:
                lib_abs = float(getattr(self.config, 'liberal_min_abs_score', 0.05))
                lib_mtf = float(getattr(self.config, 'liberal_min_mtf', 0.50))
                if abs(float(w_score)) >= lib_abs and float(mtf_score) >= lib_mtf:
                    liberal_direction = "BUY" if float(w_score) > 0 else "SELL"
                logger.info("[LIBERAL] shadow=%s | score=%+.3f mtf=%.2f (min |s|=%.2f, mtf>=%.2f)",
                            liberal_direction, float(w_score), float(mtf_score), lib_abs, lib_mtf)
                
            except Exception:
                liberal_direction = "NEUTRAL"


            contrib = signal.get('contributions', {}) if signal else raw.get('contributions', {})
            ps = contrib.get('pivot_swipe', {}) or {}
            imb = contrib.get('imbalance', {}) or {}

            cand = Candidate(
                next_bar_time=preview_close,
                direction=direction,
                actionable=bool(eligible),
                rejection_reason=rej,
                mtf_score=mtf_score,
                breadth=breadth,
                weighted_score=w_score,
                macd_hist_slope=macd_slope,
                rsi_value=rsi_val,
                rsi_cross_up=rsi_up,
                rsi_cross_down=rsi_dn,
                pattern_used=pattern_used,
                sr_room=sr_room,
                regime=str(session_info.get('session', 'UNKNOWN')),
                confidence=float(conf) / 100.0,
                oi_signal=cand_oi_signal,
                oi_change_pct=cand_oi_chg,
                saved_at=datetime.now(),
                horizon="5m",
                pivot_swipe=ps.get('direction'),
                pivot_level=ps.get('name'),
                imbalance=imb.get('direction'),                
                liberal_direction=liberal_direction
            )



            self.hit_tracker.save_candidate(cand)
            logger.info(f"[CAL] Saved pre-close candidate {preview_close.strftime('%H:%M:%S')} "
                        f"{direction} | mtf={mtf_score:.2f} | active={breadth}/6 | "
                        f"score={w_score:+.3f} | slope={macd_slope:+.6f} | rsi={rsi_val:.1f} "
                        f"{'(upX)' if rsi_up else '(dnX)' if rsi_dn else ''} | sr={sr_room} | rej={rej or '-'}")


            # ai_dir = '-'  # observe-only phase placeholder
            # ai_prob = '-'
            # logger.info(f"[CAL] Saved pre-close candidate | "
            #             f"t={preview_close.strftime('%H:%M:%S')} | dir={direction:<12} | mtf={mtf_score:>4.2f} | "
            #             f"act={breadth}/6 | score={w_score:+.3f} | macd_slope={macd_slope:+.6f} | "
            #             f"rsi={rsi_val:>5.1f}{'(upX)' if rsi_up else '(dnX)' if rsi_dn else ''} | sr={sr_room:<8} | "
            #             f"rej={str(rej or '-'):>14} | AI={ai_dir}/{ai_prob}")


            # Optional release gate for pre-close prepared signals
            try:
                if signal and getattr(self.config, 'require_expansion_for_promotion', True):
                    if not self._release_ok(signal):
                        logger.info("[Pre-Close] Blocked: release not confirmed (expansion=False)")
                        
                        return
            except Exception:
                pass


            # If no actionable signal, keep going (calibration already captured)
            if not signal:
                logger.info("Pre-close: no actionable signal (calibration captured)")
                return

            logger.info("[SIGNAL] Pre-close: actionable candidate produced")


            # STRICTER pre-close gating (forming bar is noisy)
            try:
                mtf_score = float(signal.get('mtf_analysis', {}).get('score', 0.0))
                mtf_needed = float(getattr(self.config, 'preclose_min_mtf_score', self.config.trend_alignment_threshold))
                comp = str(signal.get('composite_signal', 'NEUTRAL')).upper()
                htf_macd_sig = str(indicators_15m.get('macd', {}).get('signal_type', indicators_15m.get('macd', {}).get('signal', 'neutral'))).lower() if indicators_15m else 'neutral'
                htf_contra = (('BUY' in comp and 'bearish' in htf_macd_sig) or
                            ('SELL' in comp and 'bullish' in htf_macd_sig))
                


                if mtf_score < mtf_needed or htf_contra:
                    log_span("[Pre-Close] Strict Gate Suppressed")
                    mtf_below = bool(mtf_score < mtf_needed)
                    reasons = []
                    if mtf_below:
                        reasons.append(f"mtf_below ({mtf_score:.2f} < {mtf_needed:.2f})")
                    if htf_contra:
                        reasons.append(f"opposing_15m_macd ({htf_macd_sig})")
                    logger.info("Reason: %s", " & ".join(reasons) if reasons else "unknown")
                    logger.info(f"Pre-close: gating → needed_mtf={mtf_needed:.2f}, got={mtf_score:.2f}, htf_macd={htf_macd_sig}")
                    return
                                
            except Exception as e:
                logger.debug(f"Pre-close strict gate skipped: {e}")
                
            # Pre-Close extreme context visibility (one-line, high-signal)
            try:
                ec = signal.get('extreme_context', {}) or {}
                rsi5 = float(indicators_5m.get('rsi', {}).get('value', 0))
                rsi15 = float(indicators_15m.get('rsi', {}).get('value', 0)) if indicators_15m else 0
                htf_price_pos = ec.get('htf_price_pos')
                if htf_price_pos is not None:
                    logger.info(f"[Pre-Close] ExtremeCtx: 15m_pos={htf_price_pos:.2f} top={ec.get('top_extreme')} bot={ec.get('bottom_extreme')} "
                                f"rsi5={rsi5:.1f} rsi15={rsi15:.1f} S/R_room={'True' if not (ec.get('top_extreme') or ec.get('bottom_extreme')) else 'False'} "
                                f"breakout={ec.get('breakout_evidence')} breakdown={ec.get('breakdown_evidence')}")
                else:
                    logger.debug("[Pre-Close] Extreme context not available yet")
            except Exception as e:
                logger.debug(f"Pre-Close extreme log skipped: {e}")



            # Mark as preview and store for the exact close handler
            signal['is_preview'] = True
            signal['bar_start_time'] = str(preview_start)
            signal['bar_close_time'] = str(preview_close)

            self._preview_cache[preview_start] = {
                'signal': signal,
                'indicators_5m': indicators_5m,
                'df_5m': df_5m
            }


            # Enhanced logging for prediction visibility

            logger.info("=" * 60) 
            logger.info(f"[Pre-Close] Prediction summary for {preview_close.strftime('%H:%M:%S')} candle") 
            logger.info(f"[Pre-Close] Signal={signal.get('composite_signal')} | Confidence={signal.get('confidence', 0):.1f}% | Score={signal.get('weighted_score', 0):.3f} | MTF={signal.get('mtf_score', 0):.2f} | Active={signal.get('active_indicators', 0)}/6") 
            logger.info("=" * 60)
            logger.info(f"   Signal: {signal.get('composite_signal')}")
            logger.info(f"   Confidence: {signal.get('confidence', 0):.1f}%")
            logger.info(f"   Score: {signal.get('weighted_score', 0):.3f}")
            logger.info(f"   MTF Score: {signal.get('mtf_score', 0):.2f}")
            logger.info(f"   Active Indicators: {signal.get('active_indicators', 0)}")
            logger.info("=" * 60)


            # Store for correctness tracking
            try:
                self.last_prediction = {
                    'time': preview_close,  # next candle open time
                    'signal': signal.get('composite_signal'),
                    'confidence': signal.get('confidence'),
                    'predicted_at': datetime.now()
                }
            except Exception:
                pass

            logger.info(f"✓ Pre-close signal prepared for {preview_start.strftime('%H:%M:%S')} → send at close")
            
        except Exception as e:
            logger.error(f"Pre-close analysis error: {e}", exc_info=True)




    
    async def run_initial_analysis(self):
        """Run analysis on existing data from persistent storage."""
        try:
            # Get data from persistence
            df_5m = self.persistence_manager.get_data("5m", 100)
            
            df_15m = self.persistence_manager.get_data("15m", 100)
            
            if df_5m.empty:
                logger.info("No historical 5m data for initial analysis")
                return
            
            logger.info(f"Running initial analysis on {len(df_5m)} 5m candles")
            
            # Calculate indicators for both timeframes
            indicators_5m = await self.technical_analysis.calculate_all_indicators(df_5m)
            indicators_15m = None
            
            if not df_15m.empty:
                # Create a temporary config with 15m parameters
                indicators_15m = await self.calculate_15m_indicators(df_15m)
            
            # Generate initial signal
            initial_signal = await self.signal_analyzer.analyze_and_generate_signal(
                indicators_5m, 
                df_5m,
                indicators_15m,
                df_15m
            )
            
            if initial_signal:
                logger.info(f"Initial signal generated: {initial_signal.get('composite_signal')}")
                
        except Exception as e:
            logger.error(f"Initial analysis error: {e}")
    
    async def on_candle_complete(self, candle: pd.DataFrame, all_candles: pd.DataFrame) -> None:
        """Handle completed candle with persistent storage update."""
        try:

            if candle.empty:
                logger.warning("Empty candle received")
                return
            
            timestamp = candle.index[0] 
            
            # Idempotency guard: skip duplicate timestamps 
            if self.last_candle_ts_5m is not None and timestamp == self.last_candle_ts_5m: 
                logger.debug(f"Duplicate 5m candle {timestamp.strftime('%H:%M:%S')} ignored") 
                return
          
            # Mark first, then log
            self.last_candle_ts_5m = timestamp
            logger.info(f"{'=' * 50}")
            logger.info(f"📊 5-Min Candle Complete: {timestamp.strftime('%H:%M:%S')}")
                

            # If a pre-close signal was prepared for this start time, send it immediately 
            preview = self._preview_cache.pop(timestamp, None) 
            

            if preview:
                logger.info("Using pre-close prepared signal (no extra analysis)")
                

                prepared = preview['signal']

                # Recompute final signal on the closed bar to compare with preview
                try:
                    indicators_5m_final = await self.technical_analysis.calculate_all_indicators(preview['df_5m'])
                    # Rebuild a current 15m snapshot
                    indicators_15m_final = None
                    try:
                        df15 = self.persistence_manager.get_data("15m", 50)
                        if not df15.empty:
                            indicators_15m_final = await self.calculate_15m_indicators(df15)
                    except Exception:
                        indicators_15m_final = None





                    # Quiet recompute: suppress INFO spam during drift-only recompute
                    with _quiet_logger('consolidated_signal_analyzer', logging.WARNING):
                        final_sig = await self.signal_analyzer.analyze_and_generate_signal(
                            indicators_5m_final,
                            preview['df_5m'],
                            indicators_15m_final,
                            df15 if 'df15' in locals() else None
                        )

                    # Drift sentinel
                    try:
                        self._compare_preview_vs_final(prepared, final_sig)
                    except Exception as e:
                        logger.debug(f"[PRECLOSE] Drift compare skipped: {e}")
                except Exception as e:
                    logger.debug(f"[PRECLOSE] Final recompute skipped: {e}")

                # If suspended, skip sending preview alerts
                if getattr(self, "_preclose_suspended", False):
                    logger.info("[PRE-CLOSE] Suspended for the day due to repeated preview→final drifts")
                    
                else:
                    if await self._should_send_alert(prepared):
                        await self.send_alert(prepared, preview['indicators_5m'], preview['df_5m'])
                        return
                    else:
                        logger.info("Prepared pre-close signal did not meet alert criteria; proceeding to full analysis")

            

            # Update persistent storage for 5-min
            self.persistence_manager.update_candle(candle, "5m")
            self.stats['candles_processed_5m'] += 1
            
            # Aggregate to 15-min if needed
            self.candle_aggregator_15m.append(candle)
            if len(self.candle_aggregator_15m) >= 3:  # 3 x 5min = 15min
                candle_15m = self._create_15m_candle(self.candle_aggregator_15m)
                if candle_15m is not None:
                    self.persistence_manager.update_candle(candle_15m, "15m")
                    self.stats['candles_processed_15m'] += 1
                    logger.info(f"📊 15-Min Candle Created: {candle_15m.index[0].strftime('%H:%M:%S')}")
                self.candle_aggregator_15m = []
            
            # Get data for analysis (fresh 15m via resample to avoid boundary lag) 
            df_5m = self.persistence_manager.get_data("5m", 100)

            # Hit-rate resolution (next-bar scoring for saved candidates)
            try:
                o = float(df_5m['open'].iloc[-1])
                c = float(df_5m['close'].iloc[-1])

                # Resolve and track for losing streak
                try:
                    o = float(df_5m['open'].iloc[-1])
                    c = float(df_5m['close'].iloc[-1])
                    
                    # Check if this resolves a recent signal
                    if hasattr(self, 'last_signal_timestamp') and self.last_signal_timestamp == timestamp:
                        move = c - o
                        last_signal_direction = getattr(self, 'last_signal_direction', 'NEUTRAL')
                        
                        was_correct = False
                        if "BUY" in last_signal_direction and move > 0:
                            was_correct = True
                        elif "SELL" in last_signal_direction and move < 0:
                            was_correct = True
                        
                        # Update losing streak
                        self.recent_trade_outcomes.append(was_correct)
                        if len(self.recent_trade_outcomes) > 10:
                            self.recent_trade_outcomes.pop(0)
                        
                        if was_correct:
                            self.losing_streak_count = max(0, self.losing_streak_count - 1)
                            logger.info(f"[LOSING-STREAK] ✅ WIN - Streak reset to {self.losing_streak_count}")
                            
                            # Check for reset after consecutive wins
                            reset_wins = int(getattr(self.config, 'losing_streak_reset_after_wins', 2))
                            recent_wins = sum(self.recent_trade_outcomes[-reset_wins:])
                            if recent_wins >= reset_wins:
                                self.losing_streak_count = 0
                                self.losing_streak_pause_until = None
                                logger.info(f"[LOSING-STREAK] 🔄 Full reset after {reset_wins} consecutive wins")
                        else:
                            self.losing_streak_count += 1
                            logger.warning(f"[LOSING-STREAK] ❌ LOSS - Streak increased to {self.losing_streak_count}")
                            
                            # Check if pause needed
                            pause_after = int(getattr(self.config, 'losing_streak_pause_after', 5))
                            if self.losing_streak_count >= pause_after and not self.losing_streak_pause_until:
                                pause_minutes = int(getattr(self.config, 'losing_streak_pause_minutes', 30))
                                self.losing_streak_pause_until = datetime.now() + timedelta(minutes=pause_minutes)
                                logger.critical(
                                    f"[LOSING-STREAK] 🛑 PAUSING TRADING for {pause_minutes} minutes "
                                    f"after {self.losing_streak_count} consecutive losses"
                                )
                                
                                # Send Telegram notification
                                if self.telegram_bot:
                                    self.telegram_bot.send_message(
                                        f"⚠️ TRADING PAUSED\n"
                                        f"Losing streak: {self.losing_streak_count}\n"
                                        f"Resuming at: {self.losing_streak_pause_until.strftime('%H:%M')}"
                                    )
                    
                    self.hit_tracker.resolve_bar(timestamp, o, c, logger)
                    
                except Exception as e:
                    logger.error(f"[LOSING-STREAK] Tracking error: {e}")


                try:
                    if hasattr(self, 'setup_stats') and self.setup_stats:
                        self.setup_stats.update_from_tail()
                        logger.info("[EVIDENCE] Tables refreshed if hitrate files changed")
                except Exception:
                    pass
                
                
            except Exception as e:
                logger.debug(f"[HR] resolve_bar error: {e}")

            # Mini bucket report twice per hour
            if timestamp.minute in (0, 30):
                self.hit_tracker.report(logger, min_samples=5)


            df_15m = pd.DataFrame()
            try:
                if not self.persistence_manager.data_5m.empty:

                    # Stable NSE quarter-hour bins: include right edge, then shift label to left edge 
                    
                    
                    df_15m_resampled = (
                        self.persistence_manager.data_5m.tail(800)
                        .resample('15min', label='left', closed='left', origin='start_day', offset='15min')
                        .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
                        .dropna()
                    )


                    # Shift labels to bin start (e.g., 14:30 → 14:15) 
                    df_15m_resampled.index = df_15m_resampled.index - pd.Timedelta(minutes=15) 
                    
                    logger.debug(f"15m resample built: {len(df_15m_resampled)} bars; " f"last={df_15m_resampled.index[-1].strftime('%H:%M:%S')}")

                    # Persist the last fully closed 15m candle so metadata/stored 15m stay in sync
                    try:
                        if not df_15m_resampled.empty:
                            last_15 = df_15m_resampled.tail(1)
                            last_15_ts = last_15.index[0]
                            if self.persistence_manager.data_15m.empty or last_15_ts not in self.persistence_manager.data_15m.index:
                                
 
                                self.persistence_manager.update_candle(last_15, "15m") 
                                
                                logger.info(f"📊 15-Min Candle Created: {last_15_ts.strftime('%H:%M:%S')}") 
                                
                                logger.info(f"Stored 15m last={self.persistence_manager.data_15m.index[-1].strftime('%H:%M:%S')} count={len(self.persistence_manager.data_15m)}") 
                                
                                logger.debug(f"Stored 15m last={self.persistence_manager.data_15m.index[-1].strftime('%H:%M:%S')} count={len(self.persistence_manager.data_15m)}")
                                
                                
                    except Exception as e:
                        logger.debug(f"Persisting resampled 15m failed (ignored): {e}")



                    df_15m = df_15m_resampled.tail(50).copy()

                # Track prediction accuracy
                if hasattr(self, 'last_prediction') and self.last_prediction:
                    # Check if previous prediction was correct
                    predicted_time = self.last_prediction.get('time')
                    if predicted_time and timestamp == predicted_time:
                        actual_movement = df_5m['close'].iloc[-1] - df_5m['open'].iloc[-1]
                        predicted_signal = self.last_prediction.get('signal', '')
                        
                        if 'BUY' in predicted_signal and actual_movement > 0:
                            logger.info(f"✅ Prediction CORRECT: Expected UP, got +{actual_movement:.2f}")
                        elif 'SELL' in predicted_signal and actual_movement < 0:
                            logger.info(f"✅ Prediction CORRECT: Expected DOWN, got {actual_movement:.2f}")
                        else:
                            logger.info(f"❌ Prediction WRONG: Expected {predicted_signal}, got {actual_movement:.2f}")

                    
            except Exception as e:
                logger.debug(f"15m resample failed, falling back to stored 15m: {e}")
                df_15m = self.persistence_manager.get_data("15m", 50)
            
            # Check if we have enough data
            if len(df_5m) < self.config.min_data_points:
                logger.info(f"⏳ Waiting for more 5m data: {len(df_5m)}/{self.config.min_data_points}")
                return
            
            # Trigger analysis
            await self.analyze_and_signal(df_5m, df_15m)
            
            # Periodic save (backup)
            if (datetime.now() - self.last_save_time).total_seconds() > self.config.auto_save_interval:
                self.persistence_manager._save_to_file("5m", self.persistence_manager.data_5m)
                self.persistence_manager._save_to_file("15m", self.persistence_manager.data_15m)
                self.last_save_time = datetime.now()
                logger.debug("Periodic data save completed")
            
        except Exception as e:
            logger.error(f"Candle processing error: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def _create_15m_candle(self, candles_5m: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Aggregate 5-min candles into 15-min candle."""
        try:
            if len(candles_5m) != 3:
                return None
            
            # Combine all candles
            combined = pd.concat(candles_5m)
            
            # Create 15-min candle
            timestamp_15m = candles_5m[0].index[0]  # Start of 15-min period
            
            candle_15m = pd.DataFrame([{
                'timestamp': timestamp_15m,
                'open': candles_5m[0]['open'].iloc[0],
                'high': combined['high'].max(),
                'low': combined['low'].min(),
                'close': candles_5m[-1]['close'].iloc[-1],
                'volume': combined['volume'].sum()
            }]).set_index('timestamp')
            
            return candle_15m
            
        except Exception as e:
            logger.error(f"15-min candle creation error: {e}")
            return None
    async def calculate_15m_indicators(self, df_15m: pd.DataFrame) -> Dict:
        """Calculate indicators with 15-min parameters."""
        try:
            # Pass "15m" as timeframe parameter
            indicators_15m = await self.technical_analysis.calculate_all_indicators(df_15m, "15m")
            return indicators_15m
            
        except Exception as e:
            logger.error(f"15m indicator calculation error: {e}")
            return {}
        
    
    async def analyze_and_signal(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> None:
        """Enhanced analysis with MTF support."""
        start_time = datetime.now()
        try:
            logger.info("🔍 Starting enhanced technical analysis...")
            
            # 1. Calculate 5-min indicators
            logger.debug("Calculating 5-min indicators...")
            indicators_5m = await self.technical_analysis.calculate_all_indicators(df_5m)
            
            if not indicators_5m:
                logger.warning("5-min indicator calculation failed")
                return
            
            # Check for pattern detection signals
            if self.pattern_detector and not df_5m.empty:
                pattern_result = self.pattern_detector.detect_patterns(df_5m)
                if pattern_result['confidence'] > 0:
                    logger.info(f"Pattern detected: {pattern_result['name']} - Signal: {pattern_result['signal']}")
                    
                # Check resistance levels
                if self.resistance_detector:
                    levels = self.resistance_detector.detect_levels(df_5m)
                    logger.debug(f"Support: {levels['nearest_support']:.2f}, Resistance: {levels['nearest_resistance']:.2f}")


            # 2. Calculate 15-min indicators if available
            indicators_15m = None
            if not df_15m.empty and len(df_15m) >= 20:
                logger.debug("Calculating 15-min indicators...")
                indicators_15m = await self.calculate_15m_indicators(df_15m)

            # Optional visibility: 15m confirmations snapshot 
            if indicators_15m: 
                core = ['ema', 'macd', 'supertrend', 'rsi']
                active_15 = 0 
                for k in core: 
                    s = str(indicators_15m.get(k, {}).get('signal', 'neutral')).lower() 
                    if s not in ('neutral', 'within'): 
                        active_15 += 1 
                logger.info(f"15m core confirmations: {active_15}/4")
            
            # Check for rapid scalping opportunities
            original_confidence = None
            if self.signal_analyzer and self.signal_analyzer.last_alert_time:
                if self.signal_analyzer.should_generate_rapid_scalping_signal(
                    df_5m, 
                    self.signal_analyzer.last_alert_time
                ):
                    logger.info("⚡ Rapid scalping signal detected - adjusting thresholds")
                    # Temporarily reduce confidence requirement for rapid signals
                    original_confidence = self.config.min_confidence
                    self.config.min_confidence = max(40, original_confidence - 20)
                    # Process will use lower threshold for this signal


            # 3. Generate and analyze signal with MTF
            logger.debug("Analyzing signal with MTF...")
            final_signal = await self.signal_analyzer.analyze_and_generate_signal(
                indicators_5m, 
                df_5m,
                indicators_15m,
                df_15m
            )
            


            if final_signal:
                self.stats['signals_generated'] += 1
                
                signal_type = final_signal.get('composite_signal', 'UNKNOWN')
                confidence = final_signal.get('confidence', 0)
                
                logger.info(f"✅ Signal Generated: {signal_type} "
                           f"(Confidence: {confidence:.1f}%)")
                
                # 4. Send alert if criteria met
                if await self._should_send_alert(final_signal):
                    await self.send_alert(final_signal, indicators_5m, df_5m)
                else:
                    logger.info("Signal did not meet alert criteria")
            else:
                logger.debug("No actionable signal generated")
            

            # Restore original confidence if it was modified
            if original_confidence is not None:
                self.config.min_confidence = original_confidence
                logger.debug(f"Restored confidence threshold to {original_confidence}%")

            # Log performance
            duration = (datetime.now() - start_time).total_seconds()
            
            # log_performance(logger, "Enhanced Signal Analysis", duration, 
            #               {'5m_candles': len(df_5m), '15m_candles': len(df_15m)})
            
            log_performance(logger, "Enhanced Signal Analysis", duration, 
                            { '5m_window': len(df_5m), '15m_window': len(df_15m), 
                             '5m_total': len(self.persistence_manager.data_5m), 
                             '15m_total': len(self.persistence_manager.data_15m), 
                             '5m_last': df_5m.index[-1].strftime('%H:%M:%S') 
                             if not df_5m.empty else None, 
                             '15m_last': df_15m.index[-1].strftime('%H:%M:%S') 
                             if not df_15m.empty else None, })
                
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            self.stats['errors'] += 1
                       
            

    async def _should_send_alert(self, signal: Dict) -> bool:
        """Determine if alert should be sent."""
        log_span("[ALERT] Eligibility Check")    
        
        # Extract signal metadata FIRST (before any gates)
        try:
            signal_type = str(signal.get('composite_signal', 'NEUTRAL'))
            confidence = float(signal.get('confidence', 0.0))
        except Exception:
            signal_type = 'NEUTRAL'
            confidence = 0.0
        
        # Losing streak protection gate (uses extracted confidence)
        if getattr(self.config, 'enable_losing_streak_protection', True):
            # Pause period check
            if self.losing_streak_pause_until and datetime.now() < self.losing_streak_pause_until:
                remaining = (self.losing_streak_pause_until - datetime.now()).total_seconds() / 60
                logger.warning(f"[LOSING-STREAK] 🚫 Trading paused for {remaining:.1f} more minutes")
                return False
        
            # Boosted threshold after N losses
            streak_threshold = int(getattr(self.config, 'losing_streak_threshold', 3))
            if self.losing_streak_count >= streak_threshold:
                confidence_boost = float(getattr(self.config, 'losing_streak_confidence_boost', 10.0))
                adjusted_min_confidence = float(self.config.min_confidence) + confidence_boost
        
                if confidence < adjusted_min_confidence:
                    logger.warning(
                        f"[LOSING-STREAK] ❌ Confidence {confidence:.1f}% < {adjusted_min_confidence:.1f}% "
                        f"(streak={self.losing_streak_count}, boost=+{confidence_boost}%)"
                    )
                    return False
        
                logger.info(
                    f"[LOSING-STREAK] ✅ Meets boosted threshold "
                    f"(conf={confidence:.1f}% >= {adjusted_min_confidence:.1f}%, streak={self.losing_streak_count})"
                )
        
                    
        try:
            signal_type = signal.get('composite_signal', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            

            # Golden-lane promotion of NEUTRAL heads-up (strict)
            if signal_type in ['NEUTRAL', 'NO_SIGNAL']:
                buy_ok = False
                sell_ok = False
                try:
                    

                    cand = str(signal.get('candidate_signal','') or '')
                    mtf = float(signal.get('mtf_score', 0.0))
                    if pd.isna(mtf) or np.isinf(mtf):
                        mtf = 0.0
                    contrib = signal.get('contributions', {}) or {}
                    slope = float(contrib.get('macd', {}).get('hist_slope', 0.0))
                    if pd.isna(slope) or np.isinf(slope):
                        slope = 0.0
                    rsi_up = bool(contrib.get('rsi', {}).get('rsi_cross_up', False))
                    rsi_dn = bool(contrib.get('rsi', {}).get('rsi_cross_down', False))
                    
                    
                    rr_val = float(signal.get('risk_reward', 0.0))
                    if pd.isna(rr_val) or np.isinf(rr_val):
                        rr_val = 0.0
                    rr_floor = float(self.config.min_risk_reward_floor)
                    taper_ok = False
                    try:
                        conf = float(signal.get('confidence', 0.0))
                        mtf_val = float(signal.get('mtf_score', 0.0))
                        wscore = float(signal.get('weighted_score', 0.0))
                        contrib = signal.get('contributions', {}) or {}
                        macd_closed = float(contrib.get('macd', {}).get('hist_slope', 0.0))
                        if (self.config.enable_rr_taper and
                            conf >= float(self.config.rr_taper_confidence_min) and
                            mtf_val >= 0.70 and
                            abs(wscore) >= float(self.config.rr_taper_burst_strength) and
                            abs(macd_closed) >= 0.10):
                            taper_ok = True
                    except Exception:
                        taper_ok = False
                    rr_ok = rr_val >= (float(self.config.rr_taper_floor) if taper_ok else rr_floor)
                    
                    




                    
                    buy_ok = (cand == 'BUY_CANDIDATE' and mtf >= 0.65 and slope > 0 and rsi_up and rr_ok)
                    sell_ok = (cand == 'SELL_CANDIDATE' and mtf >= 0.65 and slope < 0 and rsi_dn and rr_ok)
                except Exception:
                    buy_ok = sell_ok = False

                # Apply expansion release gate before returning
                try:
                    if (buy_ok or sell_ok) and getattr(self.config, 'require_expansion_for_promotion', True):
                        exp_ok = self._release_ok(signal)
                        if not exp_ok:
                            logger.info("[PROMOTE] Blocked: expansion release not confirmed")
                            
                            return False
                except Exception:
                    pass

                if buy_ok or sell_ok:
                    logger.info(f"[PROMOTE] Golden-lane neutral → actionable ({cand}, mtf={mtf:.2f}, slope={slope:+.6f}, rr_ok={rr_ok})")
                    
                    return True

                
                return False

            
            if confidence < self.config.min_confidence:
                return False
            
            # Check cooldown
            if self.signal_analyzer and hasattr(self.signal_analyzer, 'last_alert_time'):
                if self.signal_analyzer.last_alert_time:
                    elapsed = (datetime.now() - self.signal_analyzer.last_alert_time).total_seconds()
                    if elapsed < self.config.base_cooldown_seconds:
                        return False
            
            
            
            # Extra guard for LIMITED S/R room for ANY alert
            try:
                sr_room = str(signal.get('mtf_analysis', {}).get('sr_room', 'UNKNOWN'))
                if sr_room == 'LIMITED':
                    mtf = float(signal.get('mtf_score', 0.0))
                    contrib = signal.get('contributions', {}) or {}
                    atr = float(contrib.get('atr', {}).get('value', 0.0)) if isinstance(contrib.get('atr'), dict) else 0.0
                    sd = contrib.get('supply_demand', {}) or {}
                    up = sd.get('nearest_up'); dn = sd.get('nearest_dn')
                    price = float(signal.get('price', 0.0) or 0.0)
                    dist = None
                    if isinstance(up, (int, float)) and price>0:
                        dist = abs(float(up)-price)
                    if isinstance(dn, (int, float)) and price>0:
                        d2 = abs(price-float(dn)); dist = min(dist, d2) if dist is not None else d2
                    
                    if not (mtf >= 0.70 and dist is not None and atr > 0 and dist >= 0.5 * atr):
                        logger.info("[ALERT] Blocked in LIMITED room: mtf=%.2f, dist=%.2f, atr=%.2f", mtf, dist or -1.0, atr)
                        return False
            except Exception:
                pass
            
            
            
            return True
        
        
        except Exception as e:
            logger.error(f"Alert check error: {e}")
            return False
    
    def _release_ok(self, signal: Dict) -> bool:
        try:
            contrib = signal.get('contributions', {}) or {}
            exp = bool(contrib.get('_ctx_expansion', False))
            logger.info("[RELEASE] expansion=%s (from contributions)", exp)
            return exp
        except Exception as e:
            logger.debug("Release check error: %s", e)
            return False


    async def send_alert(self, signal: Dict, indicators: Dict, df: pd.DataFrame) -> None:
        """Send scalping-specific trading alert via Telegram."""
        
        try:


            ist = pytz.timezone("Asia/Kolkata") 
            now_ist = datetime.now(ist) 
            base_ts = self.last_candle_ts_5m if self.last_candle_ts_5m else (df.index[-1] if not df.empty else now_ist) 
            if getattr(base_ts, 'tzinfo', None) is None: 
                base_ts_ist = ist.localize(base_ts) 
            else: 
                base_ts_ist = base_ts.astimezone(ist) 
            
            
            # Since candle is now labeled by start time, the next candle starts at close time
            current_candle_end = base_ts_ist + timedelta(minutes=5)
            forecast_5m_start = current_candle_end
            forecast_5m_end = forecast_5m_start + timedelta(minutes=5)
            forecast_15m_end = forecast_5m_start + timedelta(minutes=15)

 
            # Define timing variables clearly
            current_candle_start = base_ts_ist
            next_candle_start = current_candle_end
            next_candle_end = next_candle_start + timedelta(minutes=5)

            # Compute current price first (used as baseline for withdraw watch)
            current_price = float(df['close'].iloc[-1]) if not df.empty else 0

            # Arm withdraw watch for the predicted candle only
            try:
                next_start = forecast_5m_start
                next_end = forecast_5m_end
                entry_price = current_price
                await self._start_withdraw_watch(signal, next_start, next_end, entry_price)
            except Exception as e:
                logger.debug(f"Withdraw watch not started: {e}")
                

            current_price = float(df['close'].iloc[-1]) if not df.empty else 0
            
            # Get scalping prediction
            next_candle = signal.get('next_candle_prediction', '')
            action = signal.get('action_recommendation', '')
            weighted_score = signal.get('weighted_score', 0)
            
            # Format scalping message
            signal_type = signal.get('composite_signal', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            active_indicators = signal.get('active_indicators', 0)
            
            
            # Pattern visibility in alert logs (if analyzer attached it)
            try:
                pat = signal.get('contributions', {}).get('pattern_top', {}) or {}
                logger.info("[ALERT] Pattern=%s (%s, %s%%)", 
                            pat.get('name','NONE'), 
                            pat.get('signal','NEUTRAL'), 
                            str(pat.get('confidence','0')))
                
            except Exception:
                pass

            
            weighted_score = signal.get('weighted_score', 0)
            next_candle = signal.get('next_candle_prediction', '')
            action = signal.get('action_recommendation', '')
            current_price = float(df['close'].iloc[-1]) if not df.empty else 0
                        
            # Determine emoji and urgency
            if 'STRONG_SELL' in signal_type:
                emoji = "🔴🔴🔴"
                urgency = "HIGH CONFIDENCE SHORT"
            elif 'SELL' in signal_type:
                emoji = "🔴"
                urgency = "SHORT OPPORTUNITY"
            elif 'STRONG_BUY' in signal_type:
                emoji = "🟢🟢🟢"
                urgency = "HIGH CONFIDENCE LONG"
            elif 'BUY' in signal_type:
                emoji = "🟢"
                urgency = "LONG OPPORTUNITY"
            else:
                emoji = "⚠️"
                urgency = "WAIT"
 

            size_mult = 0.0
            try:
                size_mult = float(signal.get('entry_exit', {}).get('size_multiplier', signal.get('size_multiplier', 0.0)))
            except Exception:
                size_mult = 0.0
            logger.info(f"[ALERT] size_multiplier (at-close): {size_mult:.2f}")


            # Now build the message with defined variables
        
            message = f"""
    {emoji} <b>[{urgency}]</b> {emoji}

    <b>FORECAST WINDOW:</b> 
    • 5m: {forecast_5m_start.strftime('%H:%M')}–{forecast_5m_end.strftime('%H:%M')} IST 
    • 15m: {forecast_5m_start.strftime('%H:%M')}–{forecast_15m_end.strftime('%H:%M')} IST
      

    <b>SCALPING PREDICTION:</b>
    {next_candle}

    <b>METRICS:</b> 
    • Timeframes: 5m base; 15m confirm — {signal.get('mtf_analysis', {}).get('description', 'MTF not checked')} 
    • Price: ₹{current_price:,.2f} 
    • Weighted Score: {weighted_score:.3f} ({'SELL bias' if weighted_score < 0 else 'BUY bias' if weighted_score > 0 else 'neutral'}) 
    • Active Indicators: {active_indicators}/6 
    • Confidence: {confidence:.1f}% 
    • Candle close: {base_ts_ist.strftime('%H:%M IST')} | Sent: {now_ist.strftime('%H:%M:%S IST')} 
    • Duration: ~{signal.get('duration_prediction', {}).get('estimated_minutes', 10)} min (e.g., {forecast_5m_start.strftime('%H:%M')}-{(forecast_5m_start + timedelta(minutes=signal.get('duration_prediction', {}).get('estimated_minutes', 10))).strftime('%H:%M')})
    • Size Multiplier: {size_mult:.2f}x


    <b>ACTION:</b>
    {action}

    """
            
            # Generate chart
            chart_path = None
            if self.config.enable_charts:
                try:
                    chart_path = await self.chart_generator.generate_comprehensive_chart(
                        df, indicators, signal
                    )
                except Exception as e:
                    logger.error(f"Chart generation failed: {e}")


            # Telemetry: OI context and top pattern (display-only)
            try:
                contrib = signal.get('contributions', {}) or {}
                oi_ctx = contrib.get('oi', {}) or {}
                pat_top = contrib.get('pattern_top', {}) or {}
                
                oi_line = f"• OI: {oi_ctx.get('signal','neutral')} (ΔOI%={float(oi_ctx.get('oi_change_pct', 0.0)):.2f})"
                pat_line = f"• Pattern: {pat_top.get('name','NONE')} • {pat_top.get('signal','NEUTRAL')} ({int(pat_top.get('confidence',0))}%)"
                
                message += f"\n<b>CONTEXT:</b>\n{oi_line}\n{pat_line}\n"
                
                logger.info("[ALERT] Context → %s | %s", oi_line, pat_line)
                
            except Exception as e:
                logger.debug(f"[ALERT] OI/pattern telemetry skipped: {e}")


            # WHY explainer (guarded): add the human-readable explanation if present
            try:
                why = signal.get('why')
                if why:
                    message += f"\n<b>WHY:</b>\n{why}\n"
                    logger.info("[ALERT] WHY → %s", why)
                    
            except Exception as e:
                logger.debug(f"[ALERT] WHY block skipped: {e}")



            # Send alert — capture success and only mark stats/last_alert_time when successful
            send_success = False
            if chart_path and Path(chart_path).exists():
                send_success = self.telegram_bot.send_photo(chart_path, message)
            else:
                send_success = self.telegram_bot.send_message(message)
            
            if send_success:
                # Update alert timestamp in signal analyzer (actual send time)
                try:
                    if self.signal_analyzer:
                        self.signal_analyzer.last_alert_time = datetime.now()
                except Exception:
                    logger.debug("Could not update signal_analyzer.last_alert_time (ignored)")
                
                self.stats['alerts_sent'] += 1
            
            
                logger.info(f"✅ At-close predictive alert sent → NEXT: {forecast_5m_start.strftime('%H:%M')}-{forecast_5m_end.strftime('%H:%M')} IST ({urgency})")
                try: 
                    self.live_alerts.append({ 'time': self.last_candle_ts_5m, 'price': current_price, 'bars_left': 3, 'mfe': 0.0, 'mae': 0.0, 'dir': 'BUY' if 'BUY' in signal_type else 'SELL' }) 
                except Exception as e: 
                    logger.debug(f"Outcome tracker init failed (ignored): {e}")
                

            else:
                logger.error("Failed to send scalping alert via Telegram")



            # Store for losing streak tracking
            self.last_signal_timestamp = next_candle_start  # The predicted candle time
            self.last_signal_direction = signal_type
            logger.info(f"[SIGNAL-TRACKING] Stored signal metadata for outcome tracking")

            
        except Exception as e:
            logger.error(f"Alert sending error: {e}")

            

    async def on_tick_received(self, tick_data: Dict):
        """Handle incoming tick data."""
        # Optional: Can log tick data if needed
        pass
    
    async def on_websocket_error(self, error: Exception):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        self.stats['errors'] += 1
    
    # async def run(self):
    #     """Main run loop."""
    #     try:
    #         if not await self.initialize():
    #             logger.error("Initialization failed")
    #             return
            
    #         # Connect WebSocket
    #         logger.info("Connecting to WebSocket...")
    #         if not await self.websocket_handler.connect():
    #             logger.error("WebSocket connection failed")
    #             return
            
    #         logger.info("Starting message processing...")
    #         # Process messages
    #         await self.websocket_handler.process_messages()

    async def run(self):
        """Main run loop."""
        try:
            if not await self.initialize():
                logger.error("Initialization failed")
                return
        
                
            logger.info("Connecting to WebSocket...")
            # Run connector+processor with auto-reconnect until shutdown
            try:
                await self.websocket_handler.run_forever()
            except Exception as e:
                logger.error(f"WebSocket run_forever error: {e}", exc_info=True)
     

        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Fatal error in run loop: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
        

    async def _start_withdraw_watch(self, signal: Dict, window_start, window_end, entry_price: float):
        """Start a short-lived monitor for the predicted candle window only."""
        try:
            if not getattr(self.config, 'withdraw_monitor_enabled', True):
                return
            if float(signal.get('confidence', 0)) < float(self.config.withdraw_confidence_min):
                logger.debug("Withdraw watch skipped: confidence below threshold")
                return
            # Arm the monitor
            asyncio.create_task(self._monitor_predicted_candle(signal, window_start, window_end, entry_price))
            logger.info(f"[WithdrawWatch] Armed for {window_start.strftime('%H:%M:%S')} - {window_end.strftime('%H:%M:%S')} (predicted window only)")
        except Exception as e:
            logger.debug(f"Withdraw watch arm failed: {e}")







    def _compare_preview_vs_final(self, preview_sig: Dict[str, Any], final_sig: Optional[Dict[str, Any]]):
        """
        Enhanced drift detection with stricter gating and suspension logic.
        Detects preview→final drift and auto-suspends if K events exceeded.
        """
        try:
            if not preview_sig or not final_sig:
                logger.debug("[PRECLOSE-DRIFT] Skipping: missing signals")
                return
            
            prev_side = str(preview_sig.get('composite_signal', 'NEUTRAL')).upper()
            fin_side = str(final_sig.get('composite_signal', 'NEUTRAL')).upper()
            prev_conf = float(preview_sig.get('confidence', 0.0))
            fin_conf = float(final_sig.get('confidence', 0.0))
            
            # Detailed logging
            logger.info("=" * 60)
            logger.info("[PRECLOSE-DRIFT] Stability Check:")
            logger.info(f"  Preview:  {prev_side} @ {prev_conf:.1f}%")
            logger.info(f"  Final:    {fin_side} @ {fin_conf:.1f}%")
            
            # Detect issues
            sign_flip = ((('BUY' in prev_side) and ('SELL' in fin_side)) or 
                        (('SELL' in prev_side) and ('BUY' in fin_side)))
            drift_pp = abs(prev_conf - fin_conf)
            
            # Check for NEUTRAL drift (signal disappeared)
            neutral_drift = (prev_side != 'NEUTRAL' and fin_side == 'NEUTRAL')
            
            logger.info(f"  Sign flip: {sign_flip}")
            logger.info(f"  Confidence drift: {drift_pp:.1f} pp")
            logger.info(f"  Neutral drift: {neutral_drift}")
            
            # Drift thresholds
            pp_thresh = float(getattr(self.config, 'preclose_drift_pp', 12.0))
            k_limit = int(getattr(self.config, 'preclose_drift_cutout_K', 3))
            
            if not hasattr(self, "_preclose_drift_events"):
                self._preclose_drift_events = 0
            if not hasattr(self, "_preclose_suspended"):
                self._preclose_suspended = False
            
            # Detect drift event
            is_drift_event = sign_flip or neutral_drift or (drift_pp > pp_thresh)
            
            if is_drift_event:
                self._preclose_drift_events += 1
                
                logger.warning(
                    f"[PRECLOSE-DRIFT] ⚠️ DRIFT EVENT #{self._preclose_drift_events}/"
                    f"{k_limit} (flip={sign_flip}, neutral={neutral_drift}, Δ={drift_pp:.1f}pp)"
                )
                
                # Check for suspension
                if self._preclose_drift_events >= k_limit and not self._preclose_suspended:
                    self._preclose_suspended = True
                    logger.critical(
                        f"[PRECLOSE-DRIFT] 🚫 SUSPENDED pre-close alerts for the session "
                        f"after {k_limit} drift events"
                    )
                    
                    # Send Telegram notification
                    if hasattr(self, 'telegram_bot') and self.telegram_bot:
                        self.telegram_bot.send_message(
                            f"⚠️ PRE-CLOSE ALERTS SUSPENDED\n\n"
                            f"Reason: {self._preclose_drift_events} drift events detected\n"
                            f"- Sign flips or confidence drifts > {pp_thresh}pp\n"
                            f"- Pre-close predictions unstable\n"
                            f"- Switching to at-close analysis only\n\n"
                            f"Will resume tomorrow"
                        )
            else:
                logger.info("[PRECLOSE-DRIFT] ✅ Stable - no significant drift detected")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"[PRECLOSE-DRIFT] Error: {e}", exc_info=True)








    async def _monitor_predicted_candle(self, signal: Dict, window_start, window_end, entry_price: float):
        """Monitor only the predicted candle; issue one Withdraw/Reduce if reversal guards trip."""
        try:
            direction_buy = 'BUY' in str(signal.get('composite_signal','')).upper()
            min_dwell = int(getattr(self.config, 'withdraw_min_dwell_sec', 10))
            check_ivl = int(getattr(self.config, 'withdraw_check_interval_sec', 5))
            adverse_pts = float(getattr(self.config, 'withdraw_adverse_points', 12.0))
            frac_tp = float(getattr(self.config, 'withdraw_adverse_pct_of_tp', 0.40))
            tp = float(signal.get('take_profit', 0))
            bad_move_tp = max(adverse_pts, abs(tp - entry_price) * frac_tp) if tp > 0 else adverse_pts

            # Delay until window_start
            while datetime.now(pytz.timezone("Asia/Kolkata")) < window_start:
                await asyncio.sleep(0.2)

            # Begin watch inside the predicted window
            ist = pytz.timezone("Asia/Kolkata")
            first_ts = datetime.now(ist)
            last_ltps: List[float] = []
            adverse_cluster = 0
            decision_sent = False
            reasons: List[str] = []  # Collect reasons in ASCII

            while datetime.now(ist) <= window_end:
                await asyncio.sleep(check_ivl)

                # Pull recent ticks during this window
                try:
                    tail = self.websocket_handler.tick_buffer[-50:] if self.websocket_handler and self.websocket_handler.tick_buffer else []
                    ticks = [t for t in tail if t.get('timestamp') and window_start <= t['timestamp'] <= window_end]
                except Exception:
                    ticks = []

                if not ticks:
                    continue


                ltp = float(ticks[-1].get('ltp', entry_price))
                last_ltps.append(ltp)
                if len(last_ltps) > 20:
                    last_ltps = last_ltps[-20:]

                move = ltp - entry_price
                adverse_now = (move < 0) if direction_buy else (move > 0)
                if adverse_now:
                    adverse_cluster += 1
                    logger.debug(f"Adverse tick: cluster={adverse_cluster}, move={move:.2f}")
                else:
                    adverse_cluster = max(0, adverse_cluster - 1)
                    logger.debug(f"Non-adverse tick: cluster={adverse_cluster}, move={move:.2f}")

                # Subtle reversal: 3 consecutive adverse ticks
                try:
                    if adverse_cluster >= 2 and len(last_ltps) >= 3:
                        last3_moves = [l - entry_price for l in last_ltps[-3:]]
                        if (direction_buy and all(m < 0 for m in last3_moves)) or ((not direction_buy) and all(m > 0 for m in last3_moves)):
                            logger.info("Subtle reversal detected: 3 consecutive adverse ticks")
                            if "subtle reversal (3 adverse)" not in reasons:
                                reasons.append("subtle reversal (3 adverse)")
                except Exception:
                    pass

                # Guard 2: fast MAE breach relative to TP
                mae = abs(min(0.0, move)) if direction_buy else abs(max(0.0, move))
                mae_pts = mae
                mae_hit = mae_pts >= bad_move_tp

                # Minimal dwell before acting unless very adverse
                dwell_ok = (datetime.now(ist) - first_ts).total_seconds() >= min_dwell

                # Decision: Withdraw/Reduce (single action)
                if not decision_sent and dwell_ok and (adverse_cluster >= int(self.config.withdraw_adverse_ticks_cluster) or mae_hit):
                    local_reasons = []
                    if adverse_cluster >= int(self.config.withdraw_adverse_ticks_cluster):
                        local_reasons.append(f"adverse_ticks>={self.config.withdraw_adverse_ticks_cluster}")
                    if mae_hit:
                        local_reasons.append(f"MAE {mae_pts:.1f}>={bad_move_tp:.1f}")
                    # Merge subtle reversal reason if we saw it
                    for r in reasons:
                        if r not in local_reasons:
                            local_reasons.append(r)                    
                    
                    text = f"[WITHDRAW] {'LONG' if direction_buy else 'SHORT'}: early reversal in predicted window - " + ", ".join(local_reasons) + f" | ltp={ltp:.2f} entry={entry_price:.2f}"


                    logger.info(text)
                    try:
                        self.telegram_bot.send_message(text)
                    except Exception:
                        pass
                    decision_sent = True
                    break  # only one decision

            if not decision_sent:
                logger.info("[WithdrawWatch] Window expired: no reversal detected")
        except Exception as e:
            logger.debug(f"Withdraw watch error: {e}")


    
    
    async def shutdown(self):
        """Gracefully shutdown with data persistence."""
        logger.info("Initiating enhanced shutdown sequence...")
        self.running = False
        
        try:
            # Save final data
            if self.persistence_manager:
                logger.info("Saving final data to persistent storage...")
                self.persistence_manager._save_to_file("5m", self.persistence_manager.data_5m)
                self.persistence_manager._save_to_file("15m", self.persistence_manager.data_15m)
                self.persistence_manager.cleanup_old_data()
            
            # Calculate runtime
            runtime = datetime.now() - self.stats['start_time']
            runtime_str = str(runtime).split('.')[0]
            
            # Send shutdown notification
            if self.telegram_bot:
                shutdown_msg = f"""
                <b>🔴 System Shutdown</b>

                Runtime: {runtime_str}
                Signals: {self.stats['signals_generated']}
                Alerts: {self.stats['alerts_sent']}
                5m Candles: {self.stats['candles_processed_5m']}
                15m Candles: {self.stats['candles_processed_15m']}
                Errors: {self.stats['errors']}

                Data Saved: ✅
                """
                self.telegram_bot.send_message(shutdown_msg) 
            
            self.hit_tracker.report(logger, min_samples=3)

            # Disconnect WebSocket
            if self.websocket_handler:
                await self.websocket_handler.disconnect()
            
            logger.info("✅ Enhanced shutdown complete")
            
            # CHECK IF THIS SECTION IS ADDED CORRECTLY HERE.   
            try:
                if self._nm_task:
                    self._nm_task.cancel()
                    try:
                        await self._nm_task
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                # Weekly summary (last 7 days)
                if hasattr(self, 'setup_stats') and self.setup_stats:
                    self.setup_stats.weekly_summary(logger, days=7)
            except Exception:
                pass



            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            
                                   
# Main entry point
async def main():
    """Enhanced main entry point."""
    try:
        # Create required directories
        directories = ['logs', 'images', 'data', 'backtest']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        # Run the enhanced trading system
        system = EnhancedUnifiedTradingSystem()
        await system.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)