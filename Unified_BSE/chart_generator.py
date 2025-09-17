"""
Unified chart generation module combining all visualization capabilities.
Includes technical indicators, signals, and comprehensive analysis views.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Import pattern detector for chart annotations
try:
    from pattern_detector import CandlestickPatternDetector
    PATTERN_DETECTOR_AVAILABLE = True
except ImportError:
    PATTERN_DETECTOR_AVAILABLE = False
    logger.warning("Pattern detector not available for charts")


class UnifiedChartGenerator:
    """Comprehensive chart generator for trading signals."""
    
    def __init__(self, config):
        """Initialize chart generator."""
        self.config = config
        
        # Set style
        try:
            plt.style.use(config.chart_style)
        except:
            plt.style.use('default')
            logger.debug("Using default matplotlib style")
        
        # Color scheme
        self.colors = {
            'bullish': '#00C805',
            'bearish': '#FF3333',
            'neutral': '#888888',
            'background': '#F5F5F5',
            'grid': '#E0E0E0',
            'text': '#333333'
        }
        
        logger.info("UnifiedChartGenerator initialized")
    
    async def generate_comprehensive_chart(
        self, 
        df: pd.DataFrame, 
        indicators: Dict, 
        signal_result: Dict,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Generate comprehensive trading chart with all indicators and signals."""
        try:
            logger.info(f"Generating comprehensive chart with {len(df)} candles")
            
            # Prepare output path
            if not save_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f"{self.config.chart_save_path}/signal_{timestamp}.png"
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            fig.patch.set_facecolor(self.colors['background'])
            
            # Define layout
            gs = gridspec.GridSpec(
                6, 2, 
                height_ratios=[3, 1, 1, 1, 1, 1],
                width_ratios=[3, 1],
                hspace=0.3,
                wspace=0.2
            )
            
            # Main price chart
            ax_price = plt.subplot(gs[0, 0])
            
            # Volume chart
            ax_volume = plt.subplot(gs[1, 0], sharex=ax_price)
            
            # RSI chart
            ax_rsi = plt.subplot(gs[2, 0], sharex=ax_price)
            
            # MACD chart
            ax_macd = plt.subplot(gs[3, 0], sharex=ax_price)
            
            # Additional indicator (Keltner/Supertrend)
            ax_extra = plt.subplot(gs[4, 0], sharex=ax_price)
            
            # Signal strength bars
            ax_strength = plt.subplot(gs[5, 0], sharex=ax_price)
            
            # Signal information panel
            ax_info = plt.subplot(gs[:3, 1])
            
            # Performance metrics panel
            ax_perf = plt.subplot(gs[3:, 1])
            
            # Plot each component
            logger.debug("Plotting price and candlesticks...")
            self._plot_price_chart(ax_price, df, indicators, signal_result)
            
            logger.debug("Plotting volume...")
            self._plot_volume_chart(ax_volume, df)
            
            logger.debug("Plotting RSI...")
            self._plot_rsi_chart(ax_rsi, indicators)
            
            logger.debug("Plotting MACD...")
            self._plot_macd_chart(ax_macd, indicators)
            
            logger.debug("Plotting additional indicators...")
            self._plot_extra_indicators(ax_extra, indicators)
            
            logger.debug("Plotting signal strength...")
            self._plot_signal_strength(ax_strength, signal_result)
            
            logger.debug("Adding signal information...")
            self._add_signal_info(ax_info, signal_result, df['close'].iloc[-1])
            
            logger.debug("Adding performance metrics...")
            self._add_performance_metrics(ax_perf, signal_result)
            
    
            
            try:
            
                sig_bar_close = signal_result.get('bar_close_time') 
                if sig_bar_close: 
                    last_ts = pd.to_datetime(sig_bar_close) 
                    title_time = last_ts.strftime('%Y-%m-%d %H:%M:%S') 
                    logger.debug(f"Chart title anchored to bar_close_time={last_ts}") 
                else: 
                    last_ts = df.index[-1] 
                    title_time = last_ts.strftime('%Y-%m-%d %H:%M:%S') 
            except Exception as e: 
                logger.debug(f"Fallback chart title time due to error: {e}") 
                title_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                
            
            title = (
                f"Unified Trading Analysis - {title_time} IST\n"
                f"Signal: {signal_result.get('composite_signal', 'UNKNOWN')} | "
                f"Confidence: {signal_result.get('confidence', 0):.1f}%"
            )
            
            fig.suptitle(title, fontsize=14, fontweight='bold', color=self.colors['text'])

            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save figure
            plt.savefig(save_path, dpi=self.config.chart_dpi, 
                       bbox_inches='tight', facecolor=self.colors['background'])
            
            # Clear memory
            plt.close(fig)
            plt.close('all')
            
            logger.info(f"Chart saved successfully: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}", exc_info=True)
            plt.close('all')
            return None
    
    def _plot_price_chart(self, ax, df: pd.DataFrame, indicators: Dict, signal_result: Dict):
        """Plot price chart with candlesticks and overlays."""
        try:
            # Limit to last N candles for clarity
            plot_df = df.tail(self.config.chart_candles_to_show)
            x = np.arange(len(plot_df))
            
            # Plot candlesticks
            for i in range(len(plot_df)):
                row = plot_df.iloc[i]
                color = self.colors['bullish'] if row['close'] >= row['open'] else self.colors['bearish']
                
                # Body
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                ax.add_patch(Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                                      facecolor=color, edgecolor=color, alpha=0.8))
                
                # Wicks
                ax.plot([i, i], [row['low'], row['high']], 
                       color=color, linewidth=0.8, alpha=0.6)
            
            # Add VWAP if available
            if 'vwap' in indicators and 'vwap_series' in indicators['vwap']:
                vwap_series = indicators['vwap']['vwap_series'].tail(len(plot_df))
                ax.plot(x, vwap_series, label='VWAP', color='blue', 
                       linewidth=2, alpha=0.7)
            
            # Add Keltner Channels if available
            if 'keltner' in indicators:
                keltner = indicators['keltner']
                if 'upper_series' in keltner:
                    upper = keltner['upper_series'].tail(len(plot_df))
                    middle = keltner['middle_series'].tail(len(plot_df))
                    lower = keltner['lower_series'].tail(len(plot_df))
                    
                    ax.plot(x, upper, 'r--', label='KC Upper', linewidth=1, alpha=0.5)
                    ax.plot(x, middle, 'b-', label='KC Middle', linewidth=1, alpha=0.5)
                    ax.plot(x, lower, 'g--', label='KC Lower', linewidth=1, alpha=0.5)
                    
                    # Fill between bands
                    ax.fill_between(x, upper, lower, alpha=0.1, color='gray')
            
            # Add Supertrend if available
            if 'supertrend' in indicators and 'supertrend_series' in indicators['supertrend']:
                st_series = indicators['supertrend']['supertrend_series'].tail(len(plot_df))
                st_direction = indicators['supertrend'].get('direction_series', pd.Series()).tail(len(plot_df))
                
                # Color based on direction
                for i in range(1, len(st_series)):
                    if not pd.isna(st_series.iloc[i]) and not pd.isna(st_series.iloc[i-1]):
                        color = self.colors['bullish'] if st_direction.iloc[i] > 0 else self.colors['bearish']
                        ax.plot([i-1, i], [st_series.iloc[i-1], st_series.iloc[i]], 
                               color=color, linewidth=2, alpha=0.8)
            
            # Add entry/exit levels
            if signal_result.get('entry_price'):
                ax.axhline(y=signal_result['entry_price'], color='orange', 
                          linestyle='-.', linewidth=1.5, alpha=0.7, label='Entry')
            
            if signal_result.get('stop_loss'):
                ax.axhline(y=signal_result['stop_loss'], color='red', 
                          linestyle=':', linewidth=1.5, alpha=0.7, label='Stop Loss')
            
            if signal_result.get('take_profit'):
                ax.axhline(y=signal_result['take_profit'], color='green', 
                          linestyle=':', linewidth=1.5, alpha=0.7, label='Take Profit')
            
            # Add signal marker
            signal_type = signal_result.get('composite_signal', '')
            if 'BUY' in signal_type:
                ax.scatter(len(plot_df)-1, plot_df['low'].iloc[-1], 
                          marker='^', s=200, color=self.colors['bullish'], 
                          edgecolor='black', linewidth=2, zorder=5)
            elif 'SELL' in signal_type:
                ax.scatter(len(plot_df)-1, plot_df['high'].iloc[-1], 
                          marker='v', s=200, color=self.colors['bearish'], 
                          edgecolor='black', linewidth=2, zorder=5)
            
            # Add pattern detection annotation (non-obstructive placement)
            if PATTERN_DETECTOR_AVAILABLE:
                try:
                    detector = CandlestickPatternDetector()
                    pattern_result = detector.detect_patterns(plot_df)
                            
                    if pattern_result.get('confidence', 0) > 0:
                        # Place label in axes coordinates at top-right, away from candles
                        pattern_text = f"Pattern: {pattern_result['name']} • {pattern_result['signal']} ({int(pattern_result['confidence'])}%)"
                        ax.text(0.98, 0.98,
                                pattern_text,
                                transform=ax.transAxes,
                                ha='right', va='top',
                                fontsize=9,
                                bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='yellow',
                                        alpha=0.6),
                                zorder=10)
                except Exception as e:
                    logger.debug(f"Pattern annotation skipped: {e}")



            # Formatting
            ax.set_ylabel('Price (₹)', fontsize=10, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.set_title("Price Action & Key Levels", fontsize=11, fontweight='bold')
            
            # Set x-axis labels
            self._format_x_axis(ax, plot_df)
            
        except Exception as e:
            logger.error(f"Price chart error: {e}")
    
    def _plot_volume_chart(self, ax, df: pd.DataFrame):
        """Plot volume bars with color coding."""
        try:
            plot_df = df.tail(self.config.chart_candles_to_show)
            x = np.arange(len(plot_df))
            
            # Color based on price movement
            colors = []
            for i in range(len(plot_df)):
                if plot_df['close'].iloc[i] >= plot_df['open'].iloc[i]:
                    colors.append(self.colors['bullish'])
                else:
                    colors.append(self.colors['bearish'])
            
            # Plot volume bars
            ax.bar(x, plot_df['volume'], color=colors, alpha=0.6, edgecolor='none')
            
            # Add volume moving average
            if len(plot_df) >= 20:
                vol_ma = plot_df['volume'].rolling(20).mean()
                ax.plot(x, vol_ma, color='blue', linewidth=1.5, 
                       alpha=0.7, label='Vol MA(20)')
            
            # Formatting
            ax.set_ylabel('Volume', fontsize=10, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.set_title("Volume", fontsize=11, fontweight='bold')
            
        except Exception as e:
            logger.error(f"Volume chart error: {e}")
    
    def _plot_rsi_chart(self, ax, indicators: Dict):
        """Plot RSI indicator with zones."""
        try:
            if 'rsi' not in indicators or 'rsi_series' not in indicators['rsi']:
                ax.text(0.5, 0.5, 'RSI Data Not Available', 
                    ha='center', va='center', transform=ax.transAxes)
                return
            
            # Get RSI parameters for 5m (default for charts)
            rsi_params = self.config.get_rsi_params("5m")
            
            rsi_series = indicators['rsi']['rsi_series'].tail(self.config.chart_candles_to_show)
            x = np.arange(len(rsi_series))
            
            # Plot RSI line
            ax.plot(x, rsi_series, label='RSI', color='purple', linewidth=2)
            
            # Add zones
            ax.axhline(y=rsi_params['overbought'], color='red', 
                    linestyle='--', alpha=0.5, label='Overbought')
            ax.axhline(y=rsi_params['oversold'], color='green',
                    linestyle='--', alpha=0.5, label='Oversold')
            ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            
            # Fill zones
            ax.fill_between(x, rsi_params['overbought'], 100, 
                        alpha=0.1, color='red')
            ax.fill_between(x, 0, rsi_params['oversold'], 
                        alpha=0.1, color='green')
            
            # Highlight current value
            current_rsi = rsi_series.iloc[-1]
            color = 'red' if current_rsi > rsi_params['overbought'] else \
                'green' if current_rsi < rsi_params['oversold'] else 'purple'
            ax.scatter(len(rsi_series)-1, current_rsi, 
                    color=color, s=100, edgecolor='black', linewidth=2, zorder=5)
            
            # Formatting
            ax.set_ylabel('RSI', fontsize=10, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.set_title(f"RSI ({rsi_params['period']})", 
                        fontsize=11, fontweight='bold')
            
        except Exception as e:
            logger.error(f"RSI chart error: {e}")
    
    def _plot_macd_chart(self, ax, indicators: Dict):
        """Plot MACD indicator."""
        try:
            if 'macd' not in indicators:
                ax.text(0.5, 0.5, 'MACD Data Not Available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            macd_data = indicators['macd']
            if all(key in macd_data for key in ['macd_series', 'signal_series', 'histogram_series']):
                macd_line = macd_data['macd_series'].tail(self.config.chart_candles_to_show)
                signal_line = macd_data['signal_series'].tail(self.config.chart_candles_to_show)
                histogram = macd_data['histogram_series'].tail(self.config.chart_candles_to_show)
                
                x = np.arange(len(macd_line))
                
                # Plot MACD and signal lines
                ax.plot(x, macd_line, label='MACD', color='blue', linewidth=1.5)
                ax.plot(x, signal_line, label='Signal', color='orange', linewidth=1.5)
                
                # Plot histogram
                colors = [self.colors['bullish'] if h > 0 else self.colors['bearish'] 
                         for h in histogram]
                ax.bar(x, histogram, color=colors, alpha=0.4, label='Histogram')
                
                # Zero line
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                
                # Formatting
                ax.set_ylabel('MACD', fontsize=10, fontweight='bold')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3, color=self.colors['grid'])
                ax.set_title("MACD", fontsize=11, fontweight='bold')
            
        except Exception as e:
            logger.error(f"MACD chart error: {e}")
    
    def _plot_extra_indicators(self, ax, indicators: Dict):
        """Plot additional indicators like Impulse MACD."""
        try:
            # Placeholder for additional indicators
            ax.text(0.5, 0.5, 'Additional Indicators', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, alpha=0.5)
            
            ax.set_ylabel('Value', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.set_title("Additional Indicators", fontsize=11, fontweight='bold')
            
        except Exception as e:
            logger.error(f"Extra indicators error: {e}")
    
    def _plot_signal_strength(self, ax, signal_result: Dict):
        """Plot signal strength breakdown by indicator."""
        try:
            
            
            contributions = signal_result.get('contributions', {}) or {} 
            # Keep only items that have a numeric 'contribution' 
            
            items = [ (k, v.get('contribution')) for k, v in contributions.items() if isinstance(v, dict) and isinstance(v.get('contribution'), (int, float)) ]
            if not items: 
                ax.text(0.5, 0.5, 'No Signal Data', ha='center', va='center', transform=ax.transAxes) 
                return


            indicators = [k for k, _ in items]
            values = [float(v) for _, v in items]

            
            # Create horizontal bars
            y_pos = np.arange(len(indicators))
            colors = [self.colors['bullish'] if v > 0 else self.colors['bearish'] 
                     for v in values]
            
            bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                x = val + (0.01 if val > 0 else -0.01)
                ha = 'left' if val > 0 else 'right'
                ax.text(x, i, f'{val:.3f}', ha=ha, va='center', fontsize=9)
            
            # Formatting
            ax.set_yticks(y_pos)
            ax.set_yticklabels(indicators)
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
            ax.set_xlabel('Signal Contribution', fontsize=10, fontweight='bold')
            ax.set_title("Signal Strength Breakdown", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x', color=self.colors['grid'])
            
        except Exception as e:
            logger.error(f"Signal strength error: {e}")
    
    def _add_signal_info(self, ax, signal_result: Dict, current_price: float):
        """Add signal information panel."""
        try:
            ax.axis('off')
            
            # Prepare text
            info_lines = [
                f"SIGNAL INFORMATION",
                f"",
                f"Type: {signal_result.get('composite_signal', 'UNKNOWN')}",
                f"Action: {signal_result.get('action', 'HOLD').upper()}",
                f"Confidence: {signal_result.get('confidence', 0):.1f}%",
                f"Score: {signal_result.get('weighted_score', 0):.3f}",
            ]
            # Show detected patterns (from analyzer’s scalping_signals)
            patterns = [p for p in signal_result.get('scalping_signals', []) if p.lower().startswith('pattern')]
            if patterns:
                info_lines.append("")
                info_lines.append("PATTERN(S)")
                for p in patterns:
                    info_lines.append(p)
            info_lines += [
                f"",
                f"LEVELS",
                f"Entry: ₹{signal_result.get('entry_price', current_price):,.2f}",
                f"Stop: ₹{signal_result.get('stop_loss', 0):,.2f}",
                f"Target: ₹{signal_result.get('take_profit', 0):,.2f}",
                f"R:R: {signal_result.get('risk_reward', 0):.2f}",
                f"",
                f"DURATION",
                f"Est: {signal_result.get('duration_prediction', {}).get('estimated_minutes', 0)} mins",
                f"Conf: {signal_result.get('duration_prediction', {}).get('confidence', 'low').upper()}"
            ]


            
            # Determine colors
            signal_type = signal_result.get('composite_signal', '')
            if 'BUY' in signal_type:
                bg_color = self.colors['bullish']
                text_color = 'white'
            elif 'SELL' in signal_type:
                bg_color = self.colors['bearish'] 
                text_color = 'white'
            else:
                bg_color = self.colors['neutral']
                text_color = 'white'
            
            # Add background
            rect = Rectangle((0.05, 0.05), 0.9, 0.9, 
                           transform=ax.transAxes,
                           facecolor=bg_color, alpha=0.2,
                           edgecolor=bg_color, linewidth=2)
            ax.add_patch(rect)
            
            # Add text
            text = '\n'.join(info_lines)
            ax.text(0.5, 0.5, text, transform=ax.transAxes,
                   ha='center', va='center', fontsize=10,
                   fontfamily='monospace', fontweight='bold')
            
        except Exception as e:
            logger.error(f"Signal info error: {e}")
    
    def _add_performance_metrics(self, ax, signal_result: Dict):
        """Add performance metrics panel."""
        try:
            ax.axis('off')
            
            # Get accuracy metrics
            metrics = signal_result.get('accuracy_metrics', {})
            market_structure = signal_result.get('market_structure', {})
            
            # Prepare text
            metric_lines = [
                f"PERFORMANCE METRICS",
                f"",
                f"Signal Accuracy: {metrics.get('signal_accuracy', 0):.1f}%",
                f"Win Rate: {metrics.get('win_rate', 0):.1f}%",
                f"Confidence Sustain: {metrics.get('confidence_sustain', 0):.1f}%",
                f"Avg Profit: {metrics.get('avg_profit', 0):.2f}%",
                f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
                f"Total Trades: {metrics.get('total_trades', 0)}",
                f"",
                f"MARKET STRUCTURE",
                f"Trend: {market_structure.get('trend', 'unknown').upper()}",
                f"Strength: {market_structure.get('trend_strength', 0):.1f}%",
                f"Support: ₹{market_structure.get('support', 0):,.2f}",
                f"Resistance: ₹{market_structure.get('resistance', 0):,.2f}",
                f"Pivot: ₹{market_structure.get('pivot', 0):,.2f}"
            ]
            
            # Add background
            rect = Rectangle((0.05, 0.05), 0.9, 0.9, 
                           transform=ax.transAxes,
                           facecolor='lightblue', alpha=0.1,
                           edgecolor='blue', linewidth=1)
            ax.add_patch(rect)
            
            # Add text
            text = '\n'.join(metric_lines)
            ax.text(0.5, 0.5, text, transform=ax.transAxes,
                   ha='center', va='center', fontsize=9,
                   fontfamily='monospace')
            
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
    
    def _format_x_axis(self, ax, df: pd.DataFrame):
        """Format x-axis with time labels."""
        try:
            if hasattr(df.index, 'strftime'):
                num_labels = min(10, len(df))
                step = max(1, len(df) // num_labels)
                x_positions = np.arange(0, len(df), step)
                x_labels = [df.index[i].strftime('%H:%M') for i in x_positions]
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45)
            
        except Exception as e:
            logger.error(f"X-axis formatting error: {e}")
