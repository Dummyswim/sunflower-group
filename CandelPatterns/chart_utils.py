"""
Enhanced chart generation utilities for pattern recognition system.
Provides professional candlestick charts with pattern annotations.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Professional chart generator for pattern analysis and technical indicators.
    """
    
    def __init__(self):
        """Initialize chart generator with default settings."""
        self.colors = {
            'bullish': '#00b060',
            'bearish': '#ff3030',
            'neutral': '#808080',
            'ma_short': '#2196F3',
            'ma_medium': '#FF9800',
            'ma_long': '#9C27B0',
            'volume_up': '#00b060',
            'volume_down': '#ff3030',
            'grid': '#e0e0e0'
        }
       
        # FIX: Use proper style configuration
        try:
            # Try modern style name first
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                # Fallback to older style name
                plt.style.use('seaborn-whitegrid')
            except:
                # Use default if seaborn not available
                plt.style.use('default')
                logger.warning("Seaborn style not available, using default")
        
        # Set matplotlib parameters for better rendering
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = '#cccccc'
        plt.rcParams['grid.color'] = '#e0e0e0'
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.linewidth'] = 0.5
        
            
    def create_pattern_chart(self, df: pd.DataFrame, patterns: List[Dict],
                            prediction: Dict, indicators: Dict,
                            output_path: str, last_n_bars: int = 50) -> bool:
        """
        Create comprehensive pattern analysis chart.
        
        Args:
            df: OHLC DataFrame
            patterns: Detected patterns
            prediction: Prediction data
            indicators: Technical indicators data
            output_path: Path to save chart
            last_n_bars: Number of bars to display
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data
            df_plot = self._prepare_data(df, last_n_bars)
            if df_plot.empty:
                logger.warning("No data to plot")
                return False
            
            # Create figure with subplots
            fig = self._create_figure_layout()
            
            # Get axes
            ax_price = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
            ax_volume = plt.subplot2grid((5, 1), (3, 0), rowspan=1, sharex=ax_price)
            ax_patterns = plt.subplot2grid((5, 1), (4, 0), rowspan=1, sharex=ax_price)
            
            # Plot components
            self._plot_candlesticks(ax_price, df_plot)
            self._plot_moving_averages(ax_price, df_plot)
            self._add_pattern_markers(ax_price, df_plot, patterns)
            self._plot_volume(ax_volume, df_plot)
            self._plot_pattern_strength(ax_patterns, df_plot, patterns)
            
            # Add annotations
            self._add_prediction_box(ax_price, prediction)
            self._add_indicators_box(ax_price, indicators)
            
            # Style and save
            self._finalize_chart(fig, [ax_price, ax_volume, ax_patterns],
                               patterns, prediction)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Chart creation failed: {e}", exc_info=True)
            return self._create_fallback_chart(df, output_path)
    
    def create_performance_chart(self, prediction_history: List[Dict],
                                output_path: str) -> bool:
        """
        Create performance analysis chart showing prediction accuracy over time.
        
        Args:
            prediction_history: List of prediction results
            output_path: Path to save chart
            
        Returns:
            True if successful
        """
        try:
            if not prediction_history:
                return False
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Accuracy over time
            self._plot_accuracy_timeline(axes[0, 0], prediction_history)
            
            # Confidence vs accuracy
            self._plot_confidence_accuracy(axes[0, 1], prediction_history)
            
            # Direction performance
            self._plot_direction_performance(axes[1, 0], prediction_history)
            
            # Win/loss distribution
            self._plot_win_loss_distribution(axes[1, 1], prediction_history)
            
            plt.suptitle('Prediction Performance Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Performance chart creation failed: {e}")
            return False
    
    def _prepare_data(self, df: pd.DataFrame, last_n_bars: int) -> pd.DataFrame:
        """Prepare data for plotting."""
        if len(df) > last_n_bars:
            return df.tail(last_n_bars).copy()
        return df.copy()
    
    def _create_figure_layout(self) -> plt.Figure:
        """Create figure with proper layout."""
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('white')
        return fig
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """Plot candlestick chart."""
        x = np.arange(len(df))
        
        for i in range(len(df)):
            # Determine color
            is_bullish = df['close'].iloc[i] >= df['open'].iloc[i]
            color = self.colors['bullish'] if is_bullish else self.colors['bearish']
            
            # Draw wick
            ax.plot([x[i], x[i]], 
                   [df['low'].iloc[i], df['high'].iloc[i]],
                   color=color, linewidth=1.0, alpha=0.8)
            
            # Draw body
            body_height = abs(df['close'].iloc[i] - df['open'].iloc[i])
            body_bottom = min(df['open'].iloc[i], df['close'].iloc[i])
            
            # Ensure minimum body height for visibility
            if body_height < 0.001 * df['close'].iloc[i]:
                body_height = 0.001 * df['close'].iloc[i]
            
            rect = mpatches.Rectangle(
                (x[i] - 0.3, body_bottom), 0.6, body_height,
                facecolor=color, edgecolor=color, alpha=0.8
            )
            ax.add_patch(rect)
        
        ax.set_ylabel('Price', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.set_xlim(-1, len(df))
    
    def _plot_moving_averages(self, ax, df: pd.DataFrame):
        """Add moving averages to price chart."""
        x = np.arange(len(df))
        
        # MA periods and colors
        ma_config = [
            (10, self.colors['ma_short'], 'MA10'),
            (20, self.colors['ma_medium'], 'MA20'),
            (50, self.colors['ma_long'], 'MA50')
        ]
        
        for period, color, label in ma_config:
            if len(df) >= period:
                ma = df['close'].rolling(window=period).mean()
                ax.plot(x, ma, color=color, alpha=0.7, linewidth=1.5, label=label)
        
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    def _add_pattern_markers(self, ax, df: pd.DataFrame, patterns: List[Dict]):
        """Add pattern detection markers."""
        if not patterns or df.empty:
            return
        
        x = len(df) - 1
        price = df['close'].iloc[-1]
        
        # Sort patterns by significance
        sorted_patterns = sorted(
            patterns,
            key=lambda p: p.get('confidence', 0) * p.get('strength', 0),
            reverse=True
        )[:3]
        
        for i, pattern in enumerate(sorted_patterns):
            name = pattern['name'].replace('CDL', '')
            direction = pattern['direction']
            confidence = pattern.get('confidence', 0.5)
            
            # Determine position and color
            if direction == 'bullish':
                y_pos = price * (1.02 + i * 0.01)
                color = self.colors['bullish']
                marker = '↑'
            else:
                y_pos = price * (0.98 - i * 0.01)
                color = self.colors['bearish']
                marker = '↓'
            
            # Add annotation
            ax.annotate(
                f"{name} {marker}\n({confidence:.0%})",
                xy=(x, price),
                xytext=(x, y_pos),
                arrowprops=dict(
                    arrowstyle='->',
                    color=color,
                    lw=1.5,
                    alpha=0.7
                ),
                fontsize=9,
                color=color,
                ha='center',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                         edgecolor=color, alpha=0.8)
            )
    
    def _plot_volume(self, ax, df: pd.DataFrame):
        """Plot volume bars."""
        x = np.arange(len(df))
        volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
        
        colors = [
            self.colors['volume_up'] if df['close'].iloc[i] >= df['open'].iloc[i]
            else self.colors['volume_down']
            for i in range(len(df))
        ]
        
        ax.bar(x, volumes, color=colors, alpha=0.5, width=0.8)
        ax.set_ylabel('Volume', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.set_xlim(-1, len(df))
    
    def _plot_pattern_strength(self, ax, df: pd.DataFrame, patterns: List[Dict]):
        """Plot pattern strength/confidence bar."""
        x = np.arange(len(df))
        
        if patterns and len(df) > 0:
            # Calculate aggregate pattern strength
            avg_confidence = np.mean([p.get('confidence', 0.5) for p in patterns])
            
            # Determine color based on direction consensus
            bullish_count = sum(1 for p in patterns if p['direction'] == 'bullish')
            bearish_count = sum(1 for p in patterns if p['direction'] == 'bearish')
            
            if bullish_count > bearish_count:
                color = self.colors['bullish']
            elif bearish_count > bullish_count:
                color = self.colors['bearish']
            else:
                color = self.colors['neutral']
            
            # Plot strength bar for last candle
            ax.bar(x[-1], avg_confidence, color=color, alpha=0.7, width=0.8)
            
            # Add reference lines
            ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='High')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Neutral')
            ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, label='Low')
        
        ax.set_ylabel('Pattern Confidence', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.set_xlim(-1, len(df))
        ax.legend(loc='upper right', fontsize=8)
    
    def _add_prediction_box(self, ax, prediction: Dict):
        """Add prediction summary box."""
        if not prediction:
            return
        
        direction = prediction.get('direction', 'neutral').upper()
        confidence = prediction.get('confidence', 0)
        strength = prediction.get('strength', 'weak').upper()
        
        # Determine color
        if direction == 'BULLISH':
            color = self.colors['bullish']
            symbol = '↑'
        elif direction == 'BEARISH':
            color = self.colors['bearish']
            symbol = '↓'
        else:
            color = self.colors['neutral']
            symbol = '→'
        
        text = (
            f"{symbol} Prediction: {direction}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Strength: {strength}"
        )
        
        # Add text box
        props = dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor=color, linewidth=2, alpha=0.9)
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=props, color=color, fontweight='bold')
    
    def _add_indicators_box(self, ax, indicators: Dict):
        """Add indicators summary box."""
        if not indicators:
            return
        
        atr = indicators.get('atr', 0)
        momentum = indicators.get('momentum', 0)
        volume_trend = indicators.get('volume_profile', {}).get('volume_trend', 'neutral')
        
        text = (
            f"ATR: {atr:.2f}\n"
            f"Momentum: {momentum:.2%}\n"
            f"Volume: {volume_trend}"
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='gray', linewidth=1, alpha=0.9)
        ax.text(0.98, 0.98, text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               horizontalalignment='right',
               bbox=props, color='black')
    
    def _finalize_chart(self, fig, axes, patterns, prediction):
        """Apply final styling to chart."""
        pattern_count = len(patterns) if patterns else 0
        pred_dir = prediction.get('direction', 'neutral') if prediction else 'neutral'
        
        title = f"Pattern Analysis | {pattern_count} Pattern(s) | Prediction: {pred_dir.upper()}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Format x-axis
        ax_bottom = axes[-1]
        if hasattr(ax_bottom, 'set_xlabel'):
            ax_bottom.set_xlabel('Time Period', fontsize=11)
        
        # Remove x-labels from upper plots
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        
        plt.subplots_adjust(hspace=0.1)
    
    def _create_fallback_chart(self, df: pd.DataFrame, output_path: str) -> bool:
        """Create simple fallback chart if main chart fails."""
        try:
            plt.figure(figsize=(12, 6))
            
            if not df.empty and 'close' in df.columns:
                df['close'].plot(color='blue', linewidth=2)
                plt.title('Price Chart (Fallback)', fontsize=14)
                plt.ylabel('Price')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', fontsize=20)
            
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"Fallback chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Fallback chart failed: {e}")
            return False
    
    # Performance analysis plotting methods
    def _plot_accuracy_timeline(self, ax, history: List[Dict]):
        """Plot accuracy over time."""
        if not history:
            return
        
        # Calculate rolling accuracy
        window = min(20, len(history))
        accuracies = []
        
        for i in range(window, len(history) + 1):
            subset = history[i-window:i]
            accuracy = sum(1 for p in subset if p['correct']) / len(subset)
            accuracies.append(accuracy * 100)
        
        ax.plot(range(window, len(history) + 1), accuracies,
               color='blue', linewidth=2)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Prediction Accuracy Over Time', fontweight='bold')
        ax.set_xlabel('Prediction Number')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_accuracy(self, ax, history: List[Dict]):
        """Plot confidence vs accuracy relationship."""
        if not history:
            return
        
        # Group by confidence buckets
        buckets = {}
        for pred in history:
            conf = int(pred['confidence'] * 10) / 10  # Round to nearest 0.1
            if conf not in buckets:
                buckets[conf] = {'correct': 0, 'total': 0}
            buckets[conf]['total'] += 1
            if pred['correct']:
                buckets[conf]['correct'] += 1
        
        # Calculate accuracy for each bucket
        confidences = []
        accuracies = []
        
        for conf, data in sorted(buckets.items()):
            if data['total'] >= 3:  # Minimum sample size
                confidences.append(conf * 100)
                accuracies.append((data['correct'] / data['total']) * 100)
        
        if confidences:
            ax.scatter(confidences, accuracies, s=50, alpha=0.6)
            ax.plot(confidences, accuracies, 'b--', alpha=0.3)
        
        ax.set_title('Confidence vs Accuracy', fontweight='bold')
        ax.set_xlabel('Confidence (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_direction_performance(self, ax, history: List[Dict]):
        """Plot performance by direction."""
        if not history:
            return
        
        # Calculate stats by direction
        stats = {
            'bullish': {'correct': 0, 'total': 0},
            'bearish': {'correct': 0, 'total': 0},
            'neutral': {'correct': 0, 'total': 0}
        }
        
        for pred in history:
            direction = pred['predicted']
            if direction in stats:
                stats[direction]['total'] += 1
                if pred['correct']:
                    stats[direction]['correct'] += 1
        
        # Prepare data for plotting
        directions = []
        accuracies = []
        counts = []
        
        for direction, data in stats.items():
            if data['total'] > 0:
                directions.append(direction.capitalize())
                accuracies.append((data['correct'] / data['total']) * 100)
                counts.append(data['total'])
        
        if directions:
            bars = ax.bar(directions, accuracies, color=['green', 'red', 'gray'])
            
            # Add count labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'n={count}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Accuracy by Direction', fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_win_loss_distribution(self, ax, history: List[Dict]):
        """Plot win/loss distribution."""
        if not history:
            return
        
        wins = sum(1 for p in history if p['correct'])
        losses = len(history) - wins
        
        # Create pie chart
        sizes = [wins, losses]
        labels = [f'Wins ({wins})', f'Losses ({losses})']
        colors = [self.colors['bullish'], self.colors['bearish']]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
              startangle=90, textprops={'fontsize': 10})
        ax.set_title('Win/Loss Distribution', fontweight='bold')
