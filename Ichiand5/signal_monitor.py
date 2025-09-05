"""
Signal monitoring module - complete implementation.
Fixed version without encoding issues.
"""
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class SignalMonitor:
    """Monitor and track trading signals with health checks."""
    
    def __init__(self, config, telegram_bot):
        """Initialize signal monitor with config and telegram bot."""
        self.config = config
        self.telegram_bot = telegram_bot
        self.running = False
        self.monitor_thread = None
        self.signal_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_signals': 0,
            'strong_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'accuracy': 0.0,
            'last_update': datetime.now()
        }
        self.last_health_check = datetime.now()
        self.health_status = "HEALTHY"
        self._lock = threading.Lock()
        
        logger.info("SignalMonitor initialized")
    
    def start(self):
        """Start the signal monitoring thread."""
        if self.running:
            logger.warning("SignalMonitor already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("SignalMonitor started")
    
    def stop(self):
        """Stop the signal monitoring thread."""
        if not self.running:
            logger.warning("SignalMonitor not running")
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("SignalMonitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.last_health_check = datetime.now()
                
                self._analyze_signal_performance()
                self._check_signal_patterns()
                
                if len(self.signal_history) > 0 and len(self.signal_history) % 50 == 0:
                    self._send_performance_report()
                
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self.health_status = "ERROR"
                time.sleep(5)
    
    def add_signal(self, signal_data: Dict):
        """Add a new signal to tracking."""
        with self._lock:
            try:
                if 'timestamp' not in signal_data:
                    signal_data['timestamp'] = datetime.now()
                
                self.signal_history.append(signal_data)
                self.performance_metrics['total_signals'] += 1
                
                signal_type = signal_data.get('signal_type', '').upper()
                if 'STRONG' in signal_type:
                    self.performance_metrics['strong_signals'] += 1
                if 'BUY' in signal_type:
                    self.performance_metrics['buy_signals'] += 1
                elif 'SELL' in signal_type:
                    self.performance_metrics['sell_signals'] += 1
                
                logger.debug(f"Signal added: {signal_type}")
                
            except Exception as e:
                logger.error(f"Error adding signal: {e}")
    
    def _analyze_signal_performance(self):
        """Analyze recent signal performance."""
        with self._lock:
            try:
                if len(self.signal_history) < 10:
                    return
                
                recent_signals = list(self.signal_history)[-50:]
                
                successful = 0
                for i, signal in enumerate(recent_signals[:-5]):
                    if 'price' in signal and i + 5 < len(recent_signals):
                        future_signal = recent_signals[i + 5]
                        if 'price' in future_signal:
                            price_change = future_signal['price'] - signal['price']
                            
                            if 'BUY' in signal.get('signal_type', ''):
                                if price_change > 0:
                                    successful += 1
                            elif 'SELL' in signal.get('signal_type', ''):
                                if price_change < 0:
                                    successful += 1
                
                if len(recent_signals) > 5:
                    self.performance_metrics['successful_signals'] = successful
                    self.performance_metrics['failed_signals'] = len(recent_signals) - 5 - successful
                    self.performance_metrics['accuracy'] = (successful / (len(recent_signals) - 5)) * 100
                
                self.performance_metrics['last_update'] = datetime.now()
                
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
    
    def _check_signal_patterns(self):
        """Check for concerning signal patterns."""
        with self._lock:
            try:
                if len(self.signal_history) < 5:
                    return
                
                recent = list(self.signal_history)[-10:]
                
                signal_types = [s.get('signal_type', '') for s in recent]
                changes = sum(1 for i in range(1, len(signal_types)) 
                             if signal_types[i] != signal_types[i-1])
                
                if changes > 7:
                    logger.warning("Whipsaw pattern detected in signals")
                    if self.telegram_bot:
                        self.telegram_bot.send_message(
                            "[WARNING] Market Alert\n"
                            "Whipsaw pattern detected - signals changing rapidly.\n"
                            "Consider waiting for clearer trend."
                        )
                
            except Exception as e:
                logger.error(f"Pattern check error: {e}")
    
    def _send_performance_report(self):
        """Send performance report to Telegram."""
        try:
            metrics = self.get_metrics()
            
            message = (
                "[REPORT] Signal Performance Report\n"
                "=" * 30 + "\n"
                f"Total Signals: {metrics['total_signals']}\n"
                f"Strong Signals: {metrics['strong_signals']}\n"
                f"Buy Signals: {metrics['buy_signals']}\n"
                f"Sell Signals: {metrics['sell_signals']}\n"
                f"Accuracy: {metrics['accuracy']:.1f}%\n"
                f"Updated: {metrics['last_update'].strftime('%H:%M:%S')}"
            )
            
            if self.telegram_bot:
                self.telegram_bot.send_message(message)
            
        except Exception as e:
            logger.error(f"Performance report error: {e}")
    
    def is_healthy(self) -> bool:
        """Check if monitor is healthy."""
        try:
            if not self.running or not self.monitor_thread or not self.monitor_thread.is_alive():
                self.health_status = "STOPPED"
                return False
            
            time_since_check = (datetime.now() - self.last_health_check).seconds
            if time_since_check > 300:
                self.health_status = "STALE"
                return False
            
            self.health_status = "HEALTHY"
            return True
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self.health_status = "ERROR"
            return False
    
    def restart(self):
        """Restart the monitor."""
        logger.info("Restarting SignalMonitor")
        self.stop()
        time.sleep(1)
        self.start()
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        with self._lock:
            return self.performance_metrics.copy()
    
    def get_recent_signals(self, count: int = 10) -> List[Dict]:
        """Get recent signals."""
        with self._lock:
            return list(self.signal_history)[-count:]
    
    def get_status(self) -> Dict:
        """Get complete monitor status."""
        return {
            'health': self.health_status,
            'running': self.running,
            'last_health_check': self.last_health_check,
            'signal_count': len(self.signal_history),
            'metrics': self.get_metrics()
        }
