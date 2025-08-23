"""
Enhanced charting and alert formatting utilities with signal accuracy metrics.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import logging
import os
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

def format_enhanced_alert_with_metrics(
    price: float, 
    indicators: Dict[str, Any], 
    signal_result: Dict[str, Any], 
    timestamp: datetime
) -> str:
    """
    Format comprehensive alert message with accuracy metrics and duration prediction.
    
    Args:
        price: Current price
        indicators: Dictionary of calculated indicators
        signal_result: Signal analysis results
        timestamp: Current timestamp
        
    Returns:
        Formatted HTML message for Telegram
    """
    try:
        logger.debug("Formatting enhanced alert message")
        
        # Convert to IST timezone
        ist = pytz.timezone("Asia/Kolkata")
        timestamp_ist = timestamp.astimezone(ist)
        
        # Extract key metrics
        signal_type = signal_result.get('composite_signal', 'UNKNOWN')
        action = signal_result.get('action', 'hold').upper()
        confidence = signal_result.get('confidence', 0)
        weighted_score = signal_result.get('weighted_score', 0)
        
        # Get accuracy metrics
        accuracy_metrics = signal_result.get('accuracy_metrics', {})
        signal_accuracy = accuracy_metrics.get('signal_accuracy', 0)
        confidence_sustain = accuracy_metrics.get('confidence_sustain', 0)
        
        # Get duration prediction
        duration_pred = signal_result.get('duration_prediction', {})
        estimated_minutes = duration_pred.get('estimated_minutes', 0)
        estimated_candles = duration_pred.get('estimated_candles', 0)
        duration_confidence = duration_pred.get('confidence', 'low').upper()
        strength_trend = duration_pred.get('strength_trend', 'stable').title()
        
        # Build message without emojis
        message = f"""
<b>===== TRADING SIGNAL ALERT =====</b>

<b>[SIGNAL DETAILS]</b>
Signal Type: <b>{signal_type}</b>
Action: <b>{action}</b>
Current Price: <b>Rs {price:,.2f}</b>

<b>[SIGNAL METRICS]</b>
Confidence Score: <b>{confidence:.1f}%</b>
Weighted Score: <b>{weighted_score:.3f}</b>
Signal Accuracy: <b>{signal_accuracy:.1f}%</b>
Sustenance Confidence: <b>{confidence_sustain:.1f}%</b>

<b>[DURATION PREDICTION]</b>
Expected Duration: <b>{estimated_minutes} minutes ({estimated_candles} candles)</b>
Duration Confidence: <b>{duration_confidence}</b>
Strength Trend: <b>{strength_trend}</b>
"""
        
        # Add key levels if available
        key_levels = duration_pred.get('key_levels', {})
        if key_levels:
            message += f"""
<b>[KEY PRICE LEVELS]</b>
Resistance: <b>Rs {key_levels.get('resistance', 0):,.2f}</b>
Support: <b>Rs {key_levels.get('support', 0):,.2f}</b>
Pivot: <b>Rs {key_levels.get('pivot', 0):,.2f}</b>
"""
        
        # Add indicator breakdown
        message += "\n<b>[INDICATOR ANALYSIS]</b>\n"
        
        for name, contrib in signal_result.get('contributions', {}).items():
            indicator_signal = contrib.get('signal', 'unknown')
            contribution = contrib.get('contribution', 0)
            
            # Format indicator details based on type
            if name == "macd" and indicators.get("macd"):
                macd_val = indicators["macd"].get("macd", 0)
                signal_val = indicators["macd"].get("signal", 0)
                message += f"MACD: {macd_val:.2f} | Signal: {signal_val:.2f} | Status: {indicator_signal}\n"
                
            elif name == "rsi" and indicators.get("rsi"):
                rsi_val = indicators["rsi"].get("rsi", 50)
                message += f"RSI: {rsi_val:.1f} | Status: {indicator_signal}\n"
                
            elif name == "vwap" and indicators.get("vwap"):
                vwap_val = indicators["vwap"].get("vwap", 0)
                deviation = indicators["vwap"].get("deviation", 0)
                message += f"VWAP: Rs {vwap_val:.2f} | Deviation: {deviation:.1f}% | Status: {indicator_signal}\n"
                
            elif name == "keltner" and indicators.get("keltner"):
                position = indicators["keltner"].get("position", "unknown")
                message += f"Keltner Channels: {position} | Status: {indicator_signal}\n"
                
            elif name == "supertrend" and indicators.get("supertrend"):
                trend = indicators["supertrend"].get("trend", "unknown")
                message += f"Supertrend: {trend} | Status: {indicator_signal}\n"
                
            elif name == "impulse" and indicators.get("impulse"):
                state = indicators["impulse"].get("state", "unknown")
                message += f"Impulse System: {state} | Status: {indicator_signal}\n"
        
        # Add timestamp
        message += f"""
<b>[TIMESTAMP]</b>
Time: {timestamp_ist.strftime('%H:%M:%S IST')}
Date: {timestamp_ist.strftime('%d-%m-%Y')}

===== END OF ALERT =====
"""
        
        logger.debug("Alert message formatted successfully")
        return message
        
    except Exception as e:
        logger.error(f"Error formatting alert: {e}", exc_info=True)
        return "Alert triggered - Error formatting message"


def plot_enhanced_chart(
    ohlcv_df: pd.DataFrame, 
    indicators: Dict, 
    signal_result: Dict, 
    filename: str = "images/analysis_chart.png"
) -> bool:
    """
    Create comprehensive technical analysis chart with proper error handling.
    
    Returns:
        True if chart created successfully, False otherwise
    """
    try:
        logger.info(f"Creating enhanced chart: {filename}")
        
        # Validate input data
        if ohlcv_df.empty:
            logger.error("Empty OHLCV dataframe provided")
            return False
        
        # Set style
        plt.style.use('default')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(6, 2, height_ratios=[3, 2, 2, 2, 2, 1], width_ratios=[3, 1])
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0, 0])
        _plot_candlestick_chart(ax1, ohlcv_df)
        
        # Add technical indicators to price chart
        _add_price_indicators(ax1, indicators)
        
        ax1.set_title('Nifty50 Price Action', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (Rs)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        _plot_volume_chart(ax2, ohlcv_df)
        
        # MACD chart
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        _plot_macd_chart(ax3, indicators.get("macd"))
        
        # RSI chart
        ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
        _plot_rsi_chart(ax4, indicators.get("rsi"))
        
        # Signal strength visualization
        ax5 = fig.add_subplot(gs[4, 0])
        _plot_signal_strength(ax5, signal_result)
        
        # Signal summary panel
        ax6 = fig.add_subplot(gs[:, 1])
        _plot_signal_summary(ax6, signal_result, indicators)
        
        # Adjust layout and save
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Chart saved successfully: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Chart plotting error: {e}", exc_info=True)
        # Create fallback chart
        _create_fallback_chart(filename)
        return False


def _plot_candlestick_chart(ax, ohlcv_df):
    """Plot candlestick chart with proper error handling."""
    try:
        for idx, (timestamp, row) in enumerate(ohlcv_df.iterrows()):
            open_price = row['open']
            high = row['high']
            low = row['low']
            close = row['close']
            
            color = 'green' if close >= open_price else 'red'
            
            # Draw high-low line
            ax.plot([idx, idx], [low, high], color='black', linewidth=0.5)
            
            # Draw open-close rectangle
            height = abs(close - open_price)
            bottom = min(open_price, close)
            ax.bar(idx, height, bottom=bottom, color=color, width=0.8, alpha=0.8)
        
        # Set x-axis labels
        if len(ohlcv_df) > 0:
            step = max(1, len(ohlcv_df) // 10)
            ax.set_xticks(range(0, len(ohlcv_df), step))
            ax.set_xticklabels([
                timestamp.strftime('%H:%M') 
                for timestamp in ohlcv_df.index[::step]
            ], rotation=45)
            
    except Exception as e:
        logger.error(f"Candlestick plotting error: {e}")


def _add_price_indicators(ax, indicators):
    """Add technical indicators to price chart."""
    try:
        # Keltner Channels
        if indicators.get("keltner") and "upper_series" in indicators["keltner"]:
            keltner = indicators["keltner"]
            ax.plot(keltner["upper_series"].index, keltner["upper_series"], 
                   'r--', alpha=0.5, label='Keltner Upper')
            ax.plot(keltner["middle_series"].index, keltner["middle_series"], 
                   'b-', alpha=0.5, label='Keltner Middle')
            ax.plot(keltner["lower_series"].index, keltner["lower_series"], 
                   'g--', alpha=0.5, label='Keltner Lower')
        
        # Supertrend
        if indicators.get("supertrend") and "supertrend_series" in indicators["supertrend"]:
            st = indicators["supertrend"]
            ax.plot(st["supertrend_series"].index, st["supertrend_series"], 
                   'orange', linewidth=2, label='Supertrend')
        
        # VWAP
        if indicators.get("vwap") and "vwap_series" in indicators["vwap"]:
            ax.plot(indicators["vwap"]["vwap_series"].index, 
                   indicators["vwap"]["vwap_series"], 
                   'purple', linewidth=1.5, label='VWAP')
                   
    except Exception as e:
        logger.error(f"Error adding price indicators: {e}")


def _plot_volume_chart(ax, ohlcv_df):
    """Plot volume chart."""
    try:
        if 'volume' in ohlcv_df.columns and not ohlcv_df['volume'].empty:
            colors = ['g' if c >= o else 'r' 
                     for c, o in zip(ohlcv_df['close'], ohlcv_df['open'])]
            ax.bar(ohlcv_df.index, ohlcv_df['volume'], color=colors, alpha=0.5)
        ax.set_ylabel('Volume', fontsize=10)
        ax.grid(True, alpha=0.3)
    except Exception as e:
        logger.error(f"Volume plotting error: {e}")


def _plot_macd_chart(ax, macd_data):
    """Plot MACD chart."""
    try:
        if macd_data and "macd_series" in macd_data:
            ax.plot(macd_data["macd_series"].index, macd_data["macd_series"], 
                   'blue', label='MACD')
            ax.plot(macd_data["signal_series"].index, macd_data["signal_series"], 
                   'red', label='Signal')
            
            # Histogram
            hist = macd_data["macd_series"] - macd_data["signal_series"]
            colors = ['g' if h >= 0 else 'r' for h in hist]
            ax.bar(hist.index, hist, color=colors, alpha=0.3, label='Histogram')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('MACD', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    except Exception as e:
        logger.error(f"MACD plotting error: {e}")


def _plot_rsi_chart(ax, rsi_data):
    """Plot RSI chart."""
    try:
        if rsi_data and "rsi_series" in rsi_data:
            rsi_series = rsi_data["rsi_series"]
            ax.plot(rsi_series.index, rsi_series, 'purple', linewidth=2)
            
            # Overbought/Oversold zones
            ax.axhline(70, color='r', linestyle='--', alpha=0.5)
            ax.axhline(30, color='g', linestyle='--', alpha=0.5)
            ax.fill_between(ax.get_xlim(), 70, 100, alpha=0.1, color='red')
            ax.fill_between(ax.get_xlim(), 0, 30, alpha=0.1, color='green')
        
        ax.set_ylabel('RSI', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    except Exception as e:
        logger.error(f"RSI plotting error: {e}")


def _plot_signal_strength(ax, signal_result):
    """Plot signal strength bar chart."""
    try:
        contributions = signal_result.get('contributions', {})
        
        if contributions:
            indicators = list(contributions.keys())
            values = [contributions[ind]['contribution'] for ind in indicators]
            colors = ['green' if v > 0 else 'red' for v in values]
            
            bars = ax.barh(indicators, values, color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linewidth=1)
            ax.set_xlabel('Signal Contribution')
            ax.set_title('Indicator Contributions', fontweight='bold')
            ax.set_xlim(-0.5, 0.5)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(value, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', ha='left' if value > 0 else 'right', 
                       va='center', fontsize=8)
                       
    except Exception as e:
        logger.error(f"Signal strength plotting error: {e}")


def _plot_signal_summary(ax, signal_result, indicators):
    """Plot signal summary panel."""
    try:
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'SIGNAL SUMMARY', ha='center', fontsize=14, 
               fontweight='bold', transform=ax.transAxes)
        
        # Main signal
        signal_color = 'green' if signal_result.get('weighted_score', 0) > 0 else 'red'
        ax.text(0.5, 0.85, signal_result.get('composite_signal', 'UNKNOWN'), 
               ha='center', fontsize=16, fontweight='bold', 
               color=signal_color, transform=ax.transAxes)
        
        # Metrics
        y_pos = 0.75
        metrics = [
            f"Score: {signal_result.get('weighted_score', 0):.3f}",
            f"Confidence: {signal_result.get('confidence', 0):.1f}%",
            f"Action: {signal_result.get('action', 'HOLD').upper()}"
        ]
        
        for metric in metrics:
            ax.text(0.5, y_pos, metric, ha='center', fontsize=12, 
                   transform=ax.transAxes)
            y_pos -= 0.07
        
        # Accuracy metrics if available
        accuracy_metrics = signal_result.get('accuracy_metrics', {})
        if accuracy_metrics:
            ax.text(0.5, y_pos, 'Accuracy Metrics:', ha='center', fontsize=11, 
                   fontweight='bold', transform=ax.transAxes)
            y_pos -= 0.05
            
            ax.text(0.5, y_pos, f"Signal Accuracy: {accuracy_metrics.get('signal_accuracy', 0):.1f}%", 
                   ha='center', fontsize=10, transform=ax.transAxes)
            y_pos -= 0.04
            
            ax.text(0.5, y_pos, f"Sustenance: {accuracy_metrics.get('confidence_sustain', 0):.1f}%", 
                   ha='center', fontsize=10, transform=ax.transAxes)
            y_pos -= 0.04
        
        # Indicator status
        y_pos -= 0.02
        ax.text(0.5, y_pos, 'Indicator Status:', ha='center', fontsize=11, 
               fontweight='bold', transform=ax.transAxes)
        
        y_pos -= 0.05
        for name, contrib in signal_result.get('contributions', {}).items():
            signal = contrib.get('signal', 'unknown')
            color = 'green' if contrib.get('value', 0) > 0 else 'red' if contrib.get('value', 0) < 0 else 'gray'
            ax.text(0.5, y_pos, f"{name.upper()}: {signal}", 
                   ha='center', fontsize=9, color=color, 
                   transform=ax.transAxes)
            y_pos -= 0.04
            
    except Exception as e:
        logger.error(f"Signal summary plotting error: {e}")


def _create_fallback_chart(filename):
    """Create a simple fallback chart when main plotting fails."""
    try:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Chart Generation Error\nPlease check logs', 
                ha='center', va='center', fontsize=20)
        plt.savefig(filename)
        plt.close()
        logger.info(f"Fallback chart created: {filename}")
    except Exception as e:
        logger.error(f"Failed to create fallback chart: {e}")
