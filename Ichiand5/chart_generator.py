"""
Chart generation module for trading signals.
Fixed version with proper imports and no encoding issues.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate charts for trading signals."""
    
    def __init__(self, config):
        """Initialize chart generator with configuration."""
        self.config = config
        try:
            plt.style.use(config.CHART_STYLE)
        except:
            plt.style.use('default')
        logger.info("ChartGenerator initialized")
    
    def generate_signal_chart(self, df: pd.DataFrame, indicators: Dict, 
                            signal_result: Dict, output_path: str = None) -> Optional[str]:
        """Generate comprehensive signal chart with all indicators."""
        try:
            if df.empty:
                logger.warning("Empty dataframe for chart generation")
                return None
            
            # Create figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, 
                                    gridspec_kw={'height_ratios': [3, 1, 1, 1]})
            
            # Plot 1: Price, VWAP, and Bollinger Bands
            self._plot_price_section(axes[0], df, indicators)
            
            # Plot 2: Volume
            self._plot_volume_section(axes[1], df)
            
            # Plot 3: RSI
            self._plot_rsi_section(axes[2], indicators)
            
            # Plot 4: MACD
            self._plot_macd_section(axes[3], indicators)
            
            # Add signal annotation
            self._add_signal_annotation(fig, signal_result)
            
            # Set overall title
            fig.suptitle(f"Trading Signal Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save or display
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Chart saved to {output_path}")
                return output_path
            else:
                plt.close(fig)
                return None
                
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            plt.close('all')
            return None
    
    def _plot_price_section(self, ax, df: pd.DataFrame, indicators: Dict):
        """Plot price with VWAP and Bollinger Bands."""
        try:
            # Use last 50 points for clarity
            plot_df = df.tail(50)
            x_range = range(len(plot_df))
            
            # Plot close price
            ax.plot(x_range, plot_df['close'].values, label='Close', color='blue', linewidth=1.5)
            
            # Plot VWAP
            if 'vwap' in indicators and not indicators['vwap'].empty:
                vwap_data = indicators['vwap'].tail(50)
                ax.plot(x_range, vwap_data.values, label='VWAP', color='orange', linewidth=1, linestyle='--')
            
            # Plot Bollinger Bands
            if 'bollinger' in indicators:
                bb = indicators['bollinger']
                if 'upper' in bb:
                    upper_data = bb['upper'].tail(50)
                    ax.plot(x_range, upper_data.values, label='BB Upper', color='gray', linestyle=':', alpha=0.7)
                if 'lower' in bb:
                    lower_data = bb['lower'].tail(50)
                    ax.plot(x_range, lower_data.values, label='BB Lower', color='gray', linestyle=':', alpha=0.7)
                if 'middle' in bb:
                    middle_data = bb['middle'].tail(50)
                    ax.plot(x_range, middle_data.values, label='BB Middle', color='gray', linestyle='-', alpha=0.5)
            
            ax.set_ylabel('Price (INR)', fontsize=10)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title("Price, VWAP, and Bollinger Bands", fontsize=10)
            
        except Exception as e:
            logger.error(f"Price section plot error: {e}")
    
    def _plot_volume_section(self, ax, df: pd.DataFrame):
        """Plot volume bars."""
        try:
            plot_df = df.tail(50)
            x_range = range(len(plot_df))
            
            # Create color array based on price movement
            colors = []
            for i in range(len(plot_df)):
                if i == 0:
                    colors.append('gray')
                elif plot_df['close'].iloc[i] > plot_df['close'].iloc[i-1]:
                    colors.append('green')
                else:
                    colors.append('red')
            
            ax.bar(x_range, plot_df['volume'].values, color=colors, alpha=0.6)
            ax.set_ylabel('Volume', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_title("Volume", fontsize=10)
            
        except Exception as e:
            logger.error(f"Volume section plot error: {e}")
    
    def _plot_rsi_section(self, ax, indicators: Dict):
        """Plot RSI indicator."""
        try:
            if 'rsi' not in indicators or indicators['rsi'].empty:
                ax.text(0.5, 0.5, 'RSI Data Not Available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                return
            
            rsi_data = indicators['rsi'].tail(50)
            x_range = range(len(rsi_data))
            
            # Plot RSI line
            ax.plot(x_range, rsi_data.values, label='RSI', color='purple', linewidth=1.5)
            
            # Plot overbought/oversold lines
            ax.axhline(y=self.config.RSI_PARAMS['overbought'], color='red', linestyle='--', 
                      alpha=0.7, label='Overbought')
            ax.axhline(y=self.config.RSI_PARAMS['oversold'], color='green', linestyle='--', 
                      alpha=0.7, label='Oversold')
            ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
            
            # Fill areas
            ax.fill_between(x_range, self.config.RSI_PARAMS['overbought'], 100, 
                           alpha=0.1, color='red')
            ax.fill_between(x_range, 0, self.config.RSI_PARAMS['oversold'], 
                           alpha=0.1, color='green')
            
            ax.set_ylabel('RSI', fontsize=10)
            ax.set_ylim([0, 100])
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title("Relative Strength Index (RSI)", fontsize=10)
            
        except Exception as e:
            logger.error(f"RSI section plot error: {e}")
    
    def _plot_macd_section(self, ax, indicators: Dict):
        """Plot MACD indicator."""
        try:
            if 'macd' not in indicators or not indicators['macd']:
                ax.text(0.5, 0.5, 'MACD Data Not Available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                return
            
            macd_data = indicators['macd']
            if 'macd' in macd_data:
                macd_line = macd_data['macd'].tail(50)
                x_range = range(len(macd_line))
                ax.plot(x_range, macd_line.values, label='MACD', color='blue', linewidth=1.5)
            
            if 'signal' in macd_data:
                signal_line = macd_data['signal'].tail(50)
                ax.plot(x_range, signal_line.values, label='Signal', color='orange', linewidth=1)
            
            if 'hist' in macd_data:
                hist_data = macd_data['hist'].tail(50)
                colors = ['green' if h > 0 else 'red' for h in hist_data.values]
                ax.bar(x_range, hist_data.values, label='Histogram', color=colors, alpha=0.3)
            
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.set_ylabel('MACD', fontsize=10)
            ax.set_xlabel('Time', fontsize=10)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title("MACD", fontsize=10)
            
        except Exception as e:
            logger.error(f"MACD section plot error: {e}")
    
    def _add_signal_annotation(self, fig, signal_result: Dict):
        """Add signal information as text annotation."""
        try:
            signal_text = (
                f"Signal: {signal_result.get('composite_signal', 'UNKNOWN')}\n"
                f"Score: {signal_result.get('weighted_score', 0):.3f}\n"
                f"Confidence: {signal_result.get('confidence', 0):.1f}%\n"
                f"Active: {signal_result.get('active_indicators', 0)}/5"
            )
            
            # Determine box color based on signal
            signal_type = signal_result.get('composite_signal', '')
            if 'BUY' in signal_type:
                bbox_color = 'lightgreen'
            elif 'SELL' in signal_type:
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
