"""
Unified Trading System - Main Entry Point
Optimized with proper error handling and type checking
"""
"""
Enhanced Main Entry Point with Persistent Storage and Multi-Timeframe Support
"""
import asyncio
import sys
import signal
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


# Setup logging
setup_logging(
    logfile="logs/unified_trading.log",
    console_level=logging.INFO
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
            logger.info("✅ Configuration loaded")
            logger.debug(self.config.get_summary())
            
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
            logger.info("✅ WebSocket handler initialized")
            
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

            df_15m = pd.DataFrame()
            try:
                if not self.persistence_manager.data_5m.empty:


                    # Anchor 15m bins to NSE session start (09:15 IST) so labels are 09:15/09:30/…
                    tz = pytz.timezone('Asia/Kolkata')
                    last_ist = self.persistence_manager.data_5m.index[-1].astimezone(tz)
                    anchor = last_ist.normalize() + pd.Timedelta(hours=9, minutes=15)

                    df_15m_resampled = (
                        self.persistence_manager.data_5m
                        .resample('15min', label='left', closed='right', origin=anchor)
                        .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
                        .dropna()
                    )

                    
                    df_15m = df_15m_resampled.tail(50).copy()
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
        try:
            signal_type = signal.get('composite_signal', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            
            if signal_type in ['NEUTRAL', 'NO_SIGNAL']:
                return False
            
            if confidence < self.config.min_confidence:
                return False
            
            # Check cooldown
            if self.signal_analyzer and hasattr(self.signal_analyzer, 'last_alert_time'):
                if self.signal_analyzer.last_alert_time:
                    elapsed = (datetime.now() - self.signal_analyzer.last_alert_time).total_seconds()
                    if elapsed < self.config.base_cooldown_seconds:
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Alert check error: {e}")
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
            
            forecast_5m_start = base_ts_ist + timedelta(minutes=5) 
            forecast_5m_end = forecast_5m_start + timedelta(minutes=5) 
            forecast_15m_end = forecast_5m_start + timedelta(minutes=15) 

            
            # ist = pytz.timezone("Asia/Kolkata") 
            # timestamp_ist = datetime.now(ist)
            current_price = float(df['close'].iloc[-1]) if not df.empty else 0
            
            # Get scalping prediction
            next_candle = signal.get('next_candle_prediction', '')
            action = signal.get('action_recommendation', '')
            weighted_score = signal.get('weighted_score', 0)
            
            # Format scalping message
            signal_type = signal.get('composite_signal', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            active_indicators = signal.get('active_indicators', 0)
            
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
                emoji = "⚪"
                urgency = "WAIT"
 
      
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
                logger.info(f"Scalping alert sent: {urgency}")
                
                # Start outcome tracking for next 3 bars 
                try: 
                    self.live_alerts.append({ 'time': self.last_candle_ts_5m, 'price': current_price, 'bars_left': 3, 'mfe': 0.0, 'mae': 0.0, 'dir': 'BUY' if 'BUY' in signal_type else 'SELL' }) 
                except Exception as e: 
                    logger.debug(f"Outcome tracker init failed (ignored): {e}")

            else:
                logger.error("Failed to send scalping alert via Telegram")

            
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
    
    async def run(self):
        """Main run loop."""
        try:
            if not await self.initialize():
                logger.error("Initialization failed")
                return
            
            # Connect WebSocket
            logger.info("Connecting to WebSocket...")
            if not await self.websocket_handler.connect():
                logger.error("WebSocket connection failed")
                return
            
            logger.info("Starting message processing...")
            # Process messages
            await self.websocket_handler.process_messages()
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Fatal error in run loop: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
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
            
            # Disconnect WebSocket
            if self.websocket_handler:
                await self.websocket_handler.disconnect()
            
            logger.info("✅ Enhanced shutdown complete")
            
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