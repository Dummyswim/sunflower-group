"""
Chart generation module for technical analysis visualization.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from pathlib import Path
import io
import base64

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate technical analysis charts for alerts."""
    
    def __init__(self, config):
        """Initialize chart generator with configuration."""
        self.config = config
        plt.style.use(config.chart_style)
        
    def generate_signal_chart(self, 
                             df: pd.DataFrame, 
                             indicators: Dict,
                             signal_result: Dict,
                             output_path: str = None) -> Optional[str]:
        """
        Generate comprehensive technical analysis chart.
        
        Returns:
            Path to saved chart or None if failed
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(self.config.chart_width, self.config.chart_height))
            
            # Create grid spec for layout
            gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.05)
            
            # Main price chart with indicators
            ax1 = fig.add_subplot(gs[0])
            self._plot_price_and_indicators(ax1, df, indicators)
            
            # Volume subplot
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            self._plot_volume(ax2, df, indicators.get('obv', {}))
            
            # Stochastic subplot
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            self._plot_stochastic(ax3, indicators.get('stochastic', {}))
            
            # ADX subplot
            ax4 = fig.add_subplot(gs[3], sharex=ax1)
            self._plot_adx(ax4, indicators.get('adx', {}))
            
            # Add signal annotation
            self._add_signal_annotation(fig, signal_result)
            
            # Format x-axis
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
            plt.xticks(rotation=45)
            
            # Hide x-labels for upper subplots
            for ax in [ax1, ax2, ax3]:
                ax.set_xticklabels([])
            
            # Adjust layout
            plt.tight_layout()
            
            # Save chart
            if output_path is None:
                output_path = f"images/signal_chart_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.config.chart_dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Chart saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            plt.close()
            return None
    
    def _plot_price_and_indicators(self, ax, df: pd.DataFrame, indicators: Dict):
        """Plot price with Bollinger Bands and Ichimoku Cloud."""
        try:
            # Plot candlesticks
            last_50 = df.tail(50)
            
            for idx, (timestamp, row) in enumerate(last_50.iterrows()):
                color = 'green' if row['close'] > row['open'] else 'red'
                ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=0.5)
                height = abs(row['close'] - row['open'])
                bottom = min(row['close'], row['open'])
                rect = Rectangle((idx - 0.3, bottom), 0.6, height, 
                               facecolor=color, alpha=0.8)
                ax.add_patch(rect)
            
            # Plot Bollinger Bands
            if 'bollinger' in indicators:
                bb = indicators['bollinger']
                ax.plot(bb['upper'].tail(50).values, 'b--', alpha=0.5, label='BB Upper')
                ax.plot(bb['middle'].tail(50).values, 'b-', alpha=0.5, label='BB Middle')
                ax.plot(bb['lower'].tail(50).values, 'b--', alpha=0.5, label='BB Lower')
            
            # Plot Ichimoku Cloud
            if 'ichimoku' in indicators:
                ich = indicators['ichimoku']
                senkou_a = ich['senkou_a'].tail(50).values
                senkou_b = ich['senkou_b'].tail(50).values
                ax.fill_between(range(len(senkou_a)), senkou_a, senkou_b, 
                              where=senkou_a >= senkou_b, color='green', alpha=0.2)
                ax.fill_between(range(len(senkou_a)), senkou_a, senkou_b, 
                              where=senkou_a < senkou_b, color='red', alpha=0.2)
                ax.plot(ich['tenkan'].tail(50).values, 'orange', linewidth=1, label='Tenkan')
                ax.plot(ich['kijun'].tail(50).values, 'purple', linewidth=1, label='Kijun')
            
            ax.set_ylabel('Price (₹)')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Price plot error: {e}")
    
    def _plot_volume(self, ax, df: pd.DataFrame, obv: Dict):
        """Plot volume bars with OBV overlay."""
        try:
            volumes = df['volume'].tail(50).values
            colors = ['green' if df['close'].iloc[i] > df['open'].iloc[i] else 'red' 
                     for i in range(-50, 0)]
            
            ax.bar(range(len(volumes)), volumes, color=colors, alpha=0.5)
            
            if obv and 'obv' in obv:
                ax2 = ax.twinx()
                ax2.plot(obv['obv'].tail(50).values, 'blue', linewidth=1, label='OBV')
                ax2.plot(obv['signal'].tail(50).values, 'orange', linewidth=1, label='OBV Signal')
                ax2.set_ylabel('OBV', color='blue')
                ax2.legend(loc='upper right', fontsize=8)
            
            ax.set_ylabel('Volume')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Volume plot error: {e}")
    
    def _plot_stochastic(self, ax, stochastic: Dict):
        """Plot Stochastic Oscillator."""
        try:
            if stochastic and 'k' in stochastic:
                k = stochastic['k'].tail(50).values
                d = stochastic['d'].tail(50).values
                
                ax.plot(k, 'blue', linewidth=1, label='%K')
                ax.plot(d, 'red', linewidth=1, label='%D')
                ax.axhline(y=80, color='r', linestyle='--', alpha=0.3)
                ax.axhline(y=20, color='g', linestyle='--', alpha=0.3)
                ax.fill_between(range(len(k)), 20, 80, alpha=0.1)
                
                ax.set_ylabel('Stochastic')
                ax.set_ylim([0, 100])
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                
        except Exception as e:
            logger.error(f"Stochastic plot error: {e}")
    
    def _plot_adx(self, ax, adx: Dict):
        """Plot ADX indicator."""
        try:
            if adx and 'adx' in adx:
                adx_line = adx['adx'].tail(50).values
                plus_di = adx['plus_di'].tail(50).values
                minus_di = adx['minus_di'].tail(50).values
                
                ax.plot(adx_line, 'black', linewidth=1.5, label='ADX')
                ax.plot(plus_di, 'green', linewidth=1, label='+DI')
                ax.plot(minus_di, 'red', linewidth=1, label='-DI')
                ax.axhline(y=25, color='gray', linestyle='--', alpha=0.3)
                
                ax.set_ylabel('ADX')
                ax.set_xlabel('Time')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                
        except Exception as e:
            logger.error(f"ADX plot error: {e}")
    
    def _add_signal_annotation(self, fig, signal_result: Dict):
        """Add signal information as annotation."""
        try:
            signal_text = (
                f"Signal: {signal_result.get('composite_signal', 'UNKNOWN')}\n"
                f"Score: {signal_result.get('weighted_score', 0):.3f}\n"
                f"Confidence: {signal_result.get('confidence', 0):.1f}%\n"
                f"Active Indicators: {signal_result.get('active_indicators', 0)}/6"
            )
            
            # Determine color based on signal
            if 'BUY' in signal_result.get('composite_signal', ''):
                bbox_color = 'lightgreen'
            elif 'SELL' in signal_result.get('composite_signal', ''):
                bbox_color = 'lightcoral'
            else:
                bbox_color = 'lightgray'
            
            fig.text(0.99, 0.99, signal_text,
                    transform=fig.transFigure,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.8))
            
        except Exception as e:
            logger.error(f"Signal annotation error: {e}")
