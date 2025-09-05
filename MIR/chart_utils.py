"""
Enhanced charting and alert formatting utilities.
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
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

def format_enhanced_alert_with_duration(
    price: float, 
    indicators: Dict[str, Any], 
    signal_result: Dict[str, Any], 
    timestamp: datetime
) -> str:
    """
    Format comprehensive alert message with all required metrics.
    
    Includes:
    - Signal accuracy percentage
    - Confidence of signal sustenance
    - Duration prediction
    """
    try:
        logger.debug("Formatting enhanced alert with duration")
        
        # Convert to IST
        ist = pytz.timezone("Asia/Kolkata")
        timestamp_ist = timestamp.astimezone(ist)
        
        # Extract metrics
        signal_type = signal_result.get('composite_signal', 'UNKNOWN')
        action = signal_result.get('action', 'hold').upper()
        confidence = signal_result.get('confidence', 0)
        weighted_score = signal_result.get('weighted_score', 0)
        
        # Accuracy metrics
        accuracy_metrics = signal_result.get('accuracy_metrics', {})
        signal_accuracy = accuracy_metrics.get('signal_accuracy', 0)
        confidence_sustain = accuracy_metrics.get('confidence_sustain', 0)
        win_rate = accuracy_metrics.get('win_rate', 0)
        
        # Duration prediction
        duration_pred = signal_result.get('duration_prediction', {})
        estimated_minutes = duration_pred.get('estimated_minutes', 0)
        duration_confidence = duration_pred.get('confidence', 'low').upper()
        strength_trend = duration_pred.get('strength_trend', 'stable').title()
        
        # Entry/Exit levels
        entry_price = signal_result.get('entry_price', price)
        stop_loss = signal_result.get('stop_loss', 0)
        take_profit = signal_result.get('take_profit', 0)
        risk_reward = signal_result.get('risk_reward', 0)
        
        # Build clear alert message
        message = f"""
<b>====== NIFTY50 TRADING ALERT ======</b>

<b>[SIGNAL INFORMATION]</b>
Signal: <b>{signal_type}</b>
Action Required: <b>{action}</b>
Current Price: <b>Rs {price:,.2f}</b>
Time: <b>{timestamp_ist.strftime('%H:%M:%S IST')}</b>

<b>[ENTRY/EXIT LEVELS]</b>
Entry Price: <b>Rs {entry_price:,.2f}</b>
Stop Loss: <b>Rs {stop_loss:,.2f}</b>
Take Profit: <b>Rs {take_profit:,.2f}</b>
Risk/Reward: <b>{risk_reward:.2f}</b>

<b>[SIGNAL ACCURACY & CONFIDENCE]</b>
Signal Accuracy: <b>{signal_accuracy:.1f}%</b>
Confidence of Sustenance: <b>{confidence_sustain:.1f}%</b>
Win Rate: <b>{win_rate:.1f}%</b>
Overall Confidence: <b>{confidence:.1f}%</b>

<b>[SIGNAL DURATION PREDICTION]</b>
Expected Duration: <b>{estimated_minutes} minutes</b>
Duration Confidence: <b>{duration_confidence}</b>
Strength Trend: <b>{strength_trend}</b>

<b>[TECHNICAL INDICATORS]</b>
"""
        
        # Add indicator details
        for name, contrib in signal_result.get('contributions', {}).items():
            indicator_signal = contrib.get('signal', 'unknown')
            contribution = contrib.get('contribution', 0)
            
            if name == "macd" and indicators.get("macd"):
                macd_val = indicators["macd"].get("macd", 0)
                signal_line_val = indicators["macd"].get("signal_line", 0)  # Changed from "signal"
                histogram = indicators["macd"].get("histogram", 0)
                message += f"MACD: {macd_val:.2f} (Signal: {signal_line_val:.2f}, Hist: {histogram:.2f}) - {indicator_signal.upper()}\n"

            elif name == "rsi" and indicators.get("rsi"):
                rsi_val = indicators["rsi"].get("rsi", 50)
                message += f"RSI: {rsi_val:.1f} - {indicator_signal.upper()}\n"
                
            elif name == "vwap" and indicators.get("vwap"):
                vwap_val = indicators["vwap"].get("vwap", 0)
                deviation = indicators["vwap"].get("deviation", 0)
                if not np.isnan(vwap_val):
                    message += f"VWAP: Rs {vwap_val:.2f} (Dev: {deviation:.1f}%) - {indicator_signal.upper()}\n"
                else:
                    message += f"VWAP: No volume data - NEUTRAL\n"
                
            elif name == "keltner" and indicators.get("keltner"):
                position = indicators["keltner"].get("position", "unknown")
                upper = indicators["keltner"].get("upper", 0)
                lower = indicators["keltner"].get("lower", 0)
                message += f"Keltner: {position} (U:{upper:.0f}/L:{lower:.0f}) - {indicator_signal.upper()}\n"
                
            elif name == "supertrend" and indicators.get("supertrend"):
                trend = indicators["supertrend"].get("trend", "unknown")
                st_value = indicators["supertrend"].get("supertrend", 0)
                message += f"Supertrend: {trend} ({st_value:.2f}) - {indicator_signal.upper()}\n"
                
            elif name == "impulse" and indicators.get("impulse"):
                state = indicators["impulse"].get("state", "unknown")
                message += f"Impulse: {state} - {indicator_signal.upper()}\n"
        
        # Add market structure
        market_structure = signal_result.get('market_structure', {})
        if market_structure:
            message += f"""
<b>[MARKET STRUCTURE]</b>
Trend: <b>{market_structure.get('trend', 'unknown').upper()}</b>
Trend Strength: <b>{market_structure.get('trend_strength', 0):.1f}%</b>
"""
        
        # Add key levels
        key_levels = duration_pred.get('key_levels', {})
        if not key_levels and market_structure:
            key_levels = {
                'resistance': market_structure.get('resistance', 0),
                'support': market_structure.get('support', 0),
                'pivot': market_structure.get('pivot', 0)
            }
        
        if key_levels:
            message += f"""
<b>[KEY PRICE LEVELS]</b>
Resistance: <b>Rs {key_levels.get('resistance', 0):,.2f}</b>
Support: <b>Rs {key_levels.get('support', 0):,.2f}</b>
Pivot: <b>Rs {key_levels.get('pivot', 0):,.2f}</b>
"""
        
        # Add recommendation
        if signal_accuracy >= 70 and confidence_sustain >= 70:
            recommendation = "✅ HIGH CONFIDENCE - Consider taking position"
            emoji = "🚀"
        elif signal_accuracy >= 60 and confidence_sustain >= 60:
            recommendation = "⚠️ MODERATE CONFIDENCE - Wait for confirmation"
            emoji = "⏳"
        else:
            recommendation = "❌ LOW CONFIDENCE - Exercise caution"
            emoji = "🛑"
        
        message += f"""
<b>[RECOMMENDATION]</b>
{emoji} {recommendation}

<b>[SCORE]</b>
Weighted Score: <b>{weighted_score:.3f}</b>

====== END OF ALERT ======
"""
        
        logger.info("Alert message formatted successfully")
        return message
        
    except Exception as e:
        logger.error(f"Error formatting alert: {e}", exc_info=True)
        return "<b>Alert Error</b> - Please check logs"


def plot_enhanced_chart(
    ohlcv_df: pd.DataFrame,
    indicators: Dict[str, Any],
    signal_result: Dict[str, Any],
    output_path: str = "images/analysis_chart.png"
) -> bool:
    """
    Generate enhanced technical analysis chart with indicators and signals.
    Includes improved memory management.
    """
    try:
        logger.info(f"Generating enhanced chart with {len(ohlcv_df)} candles")
        
        # Set style (compatible version)
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            # Fallback to default if seaborn style not available
            plt.style.use('default')
            logger.debug("Using default matplotlib style")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.3)
        
        # Main price chart
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax5 = plt.subplot(gs[4], sharex=ax1)
        
        # Prepare data
        df = ohlcv_df.copy()
        if len(df) == 0:
            logger.warning("No data to plot")
            return False
            
        # Convert index to numeric for plotting
        x = np.arange(len(df))
        
        logger.debug("Plotting candlesticks")
        
        # Plot 1: Price and Candlesticks
        ax1.set_title('NIFTY50 Technical Analysis - Enhanced Signal Detection', fontsize=14, fontweight='bold')
        
        # Plot candlesticks
        for i in range(len(df)):
            color = 'green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red'
            # Body
            ax1.bar(i, df.iloc[i]['close'] - df.iloc[i]['open'], 
                   bottom=df.iloc[i]['open'], color=color, width=0.6, alpha=0.8)
            # Wicks
            ax1.plot([i, i], [df.iloc[i]['low'], df.iloc[i]['high']], 
                    color='black', linewidth=0.5, alpha=0.5)
        
        logger.debug("Adding indicators to price chart")
        
        # Add indicators on price chart
        if indicators.get("vwap") and "vwap_series" in indicators["vwap"]:
            vwap_series = indicators["vwap"]["vwap_series"]
            if len(vwap_series) == len(df):
                # Filter out NaN values for VWAP
                valid_vwap = ~vwap_series.isna()
                if valid_vwap.any():
                    ax1.plot(x[valid_vwap], vwap_series[valid_vwap], 
                            label='VWAP', color='blue', linewidth=1.5, alpha=0.7)
        
        if indicators.get("keltner"):
            keltner = indicators["keltner"]
            if "upper_series" in keltner and len(keltner["upper_series"]) == len(df):
                ax1.plot(x, keltner["upper_series"], 'r--', label='KC Upper', linewidth=0.8, alpha=0.5)
                ax1.plot(x, keltner["middle_series"], 'b-', label='KC Middle', linewidth=0.8, alpha=0.5)
                ax1.plot(x, keltner["lower_series"], 'g--', label='KC Lower', linewidth=0.8, alpha=0.5)
        
        if indicators.get("supertrend") and "supertrend_series" in indicators["supertrend"]:
            st_series = indicators["supertrend"]["supertrend_series"]
            if len(st_series) == len(df):
                valid_st = ~st_series.isna()
                if valid_st.any():
                    ax1.plot(x[valid_st], st_series[valid_st], 
                            label='Supertrend', color='purple', linewidth=2, alpha=0.8)
        
        # Add entry/exit levels
        if signal_result.get('entry_price'):
            ax1.axhline(y=signal_result['entry_price'], color='orange', 
                       linestyle='-.', linewidth=1, alpha=0.6, label='Entry')
        if signal_result.get('stop_loss'):
            ax1.axhline(y=signal_result['stop_loss'], color='red', 
                       linestyle=':', linewidth=1, alpha=0.6, label='Stop Loss')
        if signal_result.get('take_profit'):
            ax1.axhline(y=signal_result['take_profit'], color='green', 
                       linestyle=':', linewidth=1, alpha=0.6, label='Take Profit')
        
        # Add signal annotation
        signal_type = signal_result.get('composite_signal', '')
        confidence = signal_result.get('confidence', 0)
        
        # Color based on signal
        if 'BUY' in signal_type:
            signal_color = 'green'
            arrow = '↑'  # Fixed
        elif 'SELL' in signal_type:
            signal_color = 'red'
            arrow = '↓'  # Fixed
        else:
            signal_color = 'gray'
            arrow = '→'  # Fixed
        
        # Add signal box
        signal_text = f"{arrow} {signal_type}\nConfidence: {confidence:.1f}%"
        if signal_result.get('risk_reward'):
            signal_text += f"\nR:R = {signal_result['risk_reward']:.2f}"
        
        ax1.text(0.02, 0.98, signal_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=signal_color, alpha=0.3))
        
        # Add accuracy metrics
        accuracy_metrics = signal_result.get('accuracy_metrics', {})
        if accuracy_metrics:
            accuracy_text = (f"Signal Accuracy: {accuracy_metrics.get('signal_accuracy', 0):.1f}%\n"
                           f"Win Rate: {accuracy_metrics.get('win_rate', 0):.1f}%\n"
                           f"Confidence Sustain: {accuracy_metrics.get('confidence_sustain', 0):.1f}%")
            ax1.text(0.98, 0.98, accuracy_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax1.set_ylabel('Price (Rs)', fontsize=10)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        logger.debug("Plotting MACD")
        
        # Plot 2: MACD
        if indicators.get("macd"):
            macd_data = indicators["macd"]
            if "macd_series" in macd_data and len(macd_data["macd_series"]) == len(df):
                ax2.plot(x, macd_data["macd_series"], label='MACD', color='blue', linewidth=1)
                ax2.plot(x, macd_data["signal_series"], label='Signal', color='red', linewidth=1)
                
                # Histogram
                hist = macd_data["histogram_series"]
                colors = ['green' if h >= 0 else 'red' for h in hist]
                ax2.bar(x, hist, color=colors, alpha=0.3, label='Histogram')
                
                ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                ax2.set_ylabel('MACD', fontsize=10)
                ax2.legend(loc='upper left', fontsize=8)
                ax2.grid(True, alpha=0.3)
        
        logger.debug("Plotting RSI")
        
        # Plot 3: RSI
        if indicators.get("rsi") and "rsi_series" in indicators["rsi"]:
            rsi_series = indicators["rsi"]["rsi_series"]
            if len(rsi_series) == len(df):
                ax3.plot(x, rsi_series, label='RSI', color='purple', linewidth=1.5)
                ax3.axhline(y=70, color='r', linestyle='--', linewidth=0.5, alpha=0.5, label='Overbought')
                ax3.axhline(y=30, color='g', linestyle='--', linewidth=0.5, alpha=0.5, label='Oversold')
                ax3.fill_between(x, 30, 70, alpha=0.1, color='gray')
                
                # Highlight current RSI condition
                current_rsi = rsi_series.iloc[-1]
                if current_rsi >= 70:
                    ax3.scatter(x[-1], current_rsi, color='red', s=100, zorder=5)
                elif current_rsi <= 30:
                    ax3.scatter(x[-1], current_rsi, color='green', s=100, zorder=5)
                
                ax3.set_ylabel('RSI', fontsize=10)
                ax3.set_ylim(0, 100)
                ax3.legend(loc='upper left', fontsize=8)
                ax3.grid(True, alpha=0.3)
        
        logger.debug("Plotting Volume")
        
        # Plot 4: Volume
        volumes = df['volume'].values
        colors = ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red' 
                 for i in range(len(df))]
        ax4.bar(x, volumes, color=colors, alpha=0.5)
        
        # Add volume moving average
        if len(volumes) >= 20:
            vol_ma = pd.Series(volumes).rolling(20).mean()
            ax4.plot(x, vol_ma, color='blue', linewidth=1, alpha=0.5, label='Vol MA(20)')
        
        ax4.set_ylabel('Volume', fontsize=10)
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        logger.debug("Plotting indicator contributions")
        
        # Plot 5: Signal Strength & Momentum
        if signal_result.get('contributions'):
            contributions = signal_result['contributions']
            indicator_names = list(contributions.keys())
            indicator_values = [contributions[ind]['contribution'] for ind in indicator_names]
            
            colors = ['green' if v > 0 else 'red' for v in indicator_values]
            y_pos = np.arange(len(indicator_names))
            
            ax5.barh(y_pos, indicator_values, color=colors, alpha=0.6)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(indicator_names)
            ax5.set_xlabel('Signal Contribution', fontsize=10)
            ax5.set_title('Indicator Contributions', fontsize=10)
            ax5.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            ax5.grid(True, alpha=0.3)
        
        # Format x-axis
        if hasattr(df.index, 'strftime'):
            # Show limited number of labels to avoid crowding
            num_labels = min(10, len(df))
            step = max(1, len(df) // num_labels)
            tick_positions = x[::step]
            tick_labels = [df.index[i].strftime('%H:%M') for i in tick_positions]
            ax4.set_xticks(tick_positions)
            ax4.set_xticklabels(tick_labels, rotation=45)
        
        # Add duration prediction and market structure
        duration_pred = signal_result.get('duration_prediction', {})
        market_structure = signal_result.get('market_structure', {})
        
        info_text = ""
        if duration_pred:
            info_text += (f"Duration: {duration_pred.get('estimated_minutes', 0)} mins "
                         f"({duration_pred.get('confidence', 'low').upper()})\n")
        if market_structure:
            info_text += f"Market: {market_structure.get('trend', 'unknown').upper()}"
        
        if info_text:
            fig.text(0.99, 0.01, info_text, ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Overall title with timestamp
        ist = pytz.timezone("Asia/Kolkata")
        current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')
        fig.suptitle(f'Technical Analysis Report - {current_time}', fontsize=12)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Ensure directory exists (with safety check)
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if not empty string
            os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        
        # Clear figure to free memory
        fig.clf()
        plt.close('all')
        
        logger.info(f"Chart saved successfully to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Chart generation error: {e}", exc_info=True)
        # Ensure cleanup even on error
        plt.close('all')
        return False


def format_enhanced_alert(
    price: float,
    indicators: Dict[str, Any],
    signal_result: Dict[str, Any],
    timestamp: datetime
) -> str:
    """
    Backward compatibility function - redirects to format_enhanced_alert_with_duration.
    """
    logger.debug("Using backward compatibility wrapper for alert formatting")
    return format_enhanced_alert_with_duration(price, indicators, signal_result, timestamp)
