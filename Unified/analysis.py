"""
Trading System Performance Analyzer
Comprehensive analysis of trading system performance from logs
"""

import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

class TradingSystemAnalyzer:
    """Analyze trading system performance from logs and data"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.signals = []
        self.candles = []
        self.alerts = []
        self.errors = []
        self.rejections = []
        self.performance_metrics = {}
        
    def parse_log_file(self) -> Dict[str, Any]:
        """Parse log file and extract relevant information"""
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        stats = {
            'start_time': None,
            'end_time': None,
            'total_runtime': None,
            'candles_created': 0,
            'signals_generated': 0,
            'alerts_sent': 0,
            'signals_rejected': 0,
            'errors': 0,
            'signal_types': {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0},
            'rejection_reasons': {},
            'hourly_signal_rate': [],
            'confidence_scores': [],
            'price_range': {'min': float('inf'), 'max': 0},
            'indicators_used': set(),
            'packet_types_processed': {},
            'websocket_reconnections': 0,
            'telegram_failures': 0
        }
        
        for line in lines:
            # Extract timestamp
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                
                if not stats['start_time']:
                    stats['start_time'] = timestamp
                stats['end_time'] = timestamp
            
            # Parse candle creation
            if 'Candle created for' in line:
                candle_match = re.search(r'Candle created for (\d{2}:\d{2}): O:([\d.]+) H:([\d.]+) L:([\d.]+) C:([\d.]+) V:(\d+)', line)
                if candle_match:
                    stats['candles_created'] += 1
                    ohlc = {
                        'time': candle_match.group(1),
                        'open': float(candle_match.group(2)),
                        'high': float(candle_match.group(3)),
                        'low': float(candle_match.group(4)),
                        'close': float(candle_match.group(5)),
                        'volume': int(candle_match.group(6))
                    }
                    self.candles.append(ohlc)
                    stats['price_range']['min'] = min(stats['price_range']['min'], ohlc['low'])
                    stats['price_range']['max'] = max(stats['price_range']['max'], ohlc['high'])
            
            # Parse signal generation
            if 'Signal generated:' in line:
                signal_match = re.search(r'Signal generated: (\w+), Score: ([-\d.]+), Confidence: ([\d.]+)%, Active: (\d+)', line)
                if signal_match:
                    signal_type = signal_match.group(1)
                    score = float(signal_match.group(2))
                    confidence = float(signal_match.group(3))
                    active = int(signal_match.group(4))
                    
                    stats['signals_generated'] += 1
                    stats['signal_types'][signal_type] = stats['signal_types'].get(signal_type, 0) + 1
                    stats['confidence_scores'].append(confidence)
                    
                    self.signals.append({
                        'timestamp': timestamp,
                        'type': signal_type,
                        'score': score,
                        'confidence': confidence,
                        'active_indicators': active
                    })
            
            # Parse alerts sent
            if 'Alert sent:' in line:
                alert_match = re.search(r'Alert sent: (\w+) at ([\d.]+)', line)
                if alert_match:
                    stats['alerts_sent'] += 1
                    self.alerts.append({
                        'timestamp': timestamp,
                        'type': alert_match.group(1),
                        'price': float(alert_match.group(2))
                    })
            
            # Parse signal rejections
            if 'Signal rejected:' in line:
                stats['signals_rejected'] += 1
                rejection_match = re.search(r'Signal rejected: \[(.*?)\]', line)
                if rejection_match:
                    reasons = rejection_match.group(1).split(', ')
                    for reason in reasons:
                        reason = reason.strip("'")
                        stats['rejection_reasons'][reason] = stats['rejection_reasons'].get(reason, 0) + 1
                        self.rejections.append({
                            'timestamp': timestamp,
                            'reason': reason
                        })
            
            # Parse errors
            if 'ERROR' in line:
                stats['errors'] += 1
                self.errors.append({
                    'timestamp': timestamp,
                    'message': line.strip()
                })
                
                if 'Telegram' in line:
                    stats['telegram_failures'] += 1
            
            # Parse indicators
            if 'calculated:' in line:
                indicator_match = re.search(r'(\w+) calculated:', line)
                if indicator_match:
                    stats['indicators_used'].add(indicator_match.group(1))
            
            # Parse packet types
            if 'Received' in line and 'packet' in line:
                packet_match = re.search(r'Received (\w+) packet', line)
                if packet_match:
                    packet_type = packet_match.group(1)
                    stats['packet_types_processed'][packet_type] = stats['packet_types_processed'].get(packet_type, 0) + 1
            
            # Parse reconnections
            if 'Reconnection successful' in line:
                stats['websocket_reconnections'] += 1
        
        # Calculate runtime
        if stats['start_time'] and stats['end_time']:
            stats['total_runtime'] = stats['end_time'] - stats['start_time']
            runtime_hours = stats['total_runtime'].total_seconds() / 3600
            
            if runtime_hours > 0:
                stats['signals_per_hour'] = stats['signals_generated'] / runtime_hours
                stats['alerts_per_hour'] = stats['alerts_sent'] / runtime_hours
        
        # Calculate average confidence
        if stats['confidence_scores']:
            stats['avg_confidence'] = np.mean(stats['confidence_scores'])
            stats['min_confidence'] = min(stats['confidence_scores'])
            stats['max_confidence'] = max(stats['confidence_scores'])
        
        return stats
    
    def analyze_signal_patterns(self) -> Dict[str, Any]:
        """Analyze signal patterns and effectiveness"""
        
        if not self.signals:
            return {}
        
        df = pd.DataFrame(self.signals)
        
        patterns = {
            'signal_distribution': df['type'].value_counts().to_dict(),
            'avg_confidence_by_type': df.groupby('type')['confidence'].mean().to_dict(),
            'avg_score_by_type': df.groupby('type')['score'].mean().to_dict(),
            'signal_transitions': [],
            'bullish_bearish_ratio': 0,
            'signal_clustering': []
        }
        
        # Analyze signal transitions
        for i in range(1, len(df)):
            if df.iloc[i-1]['type'] != df.iloc[i]['type']:
                patterns['signal_transitions'].append({
                    'from': df.iloc[i-1]['type'],
                    'to': df.iloc[i]['type'],
                    'confidence_change': df.iloc[i]['confidence'] - df.iloc[i-1]['confidence']
                })
        
        # Calculate bullish/bearish ratio
        buy_signals = df[df['type'] == 'BUY'].shape[0]
        sell_signals = df[df['type'] == 'SELL'].shape[0]
        if sell_signals > 0:
            patterns['bullish_bearish_ratio'] = buy_signals / sell_signals
        
        # Detect signal clustering (rapid signals)
        df['time_diff'] = df['timestamp'].diff()
        rapid_signals = df[df['time_diff'] < pd.Timedelta(minutes=2)]
        if not rapid_signals.empty:
            patterns['signal_clustering'] = len(rapid_signals)
        
        return patterns
    
    def analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze market conditions from candle data"""
        
        if not self.candles:
            return {}
        
        df = pd.DataFrame(self.candles)
        
        conditions = {
            'price_volatility': df['close'].std(),
            'avg_range': (df['high'] - df['low']).mean(),
            'trend_direction': 'neutral',
            'support_levels': [],
            'resistance_levels': [],
            'price_momentum': 0
        }
        
        # Determine trend
        if len(df) > 20:
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > sma_20 * 1.01:
                conditions['trend_direction'] = 'bullish'
            elif current_price < sma_20 * 0.99:
                conditions['trend_direction'] = 'bearish'
        
        # Calculate momentum
        if len(df) > 10:
            conditions['price_momentum'] = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100
        
        # Identify support/resistance
        conditions['support_levels'] = df['low'].nsmallest(3).tolist()
        conditions['resistance_levels'] = df['high'].nlargest(3).tolist()
        
        return conditions
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        stats = self.parse_log_file()
        patterns = self.analyze_signal_patterns()
        conditions = self.analyze_market_conditions()
        
        # Calculate efficiency metrics
        if stats['signals_generated'] > 0:
            alert_efficiency = (stats['alerts_sent'] / stats['signals_generated']) * 100
            rejection_rate = (stats['signals_rejected'] / stats['signals_generated']) * 100
        else:
            alert_efficiency = 0
            rejection_rate = 0
        
        report = f"""
================================================================================
                    TRADING SYSTEM PERFORMANCE ANALYSIS REPORT
================================================================================

EXECUTION SUMMARY
-----------------
Start Time: {stats.get('start_time', 'N/A')}
End Time: {stats.get('end_time', 'N/A')}
Total Runtime: {stats.get('total_runtime', 'N/A')}
System Efficiency: {alert_efficiency:.1f}%

SIGNAL GENERATION METRICS
-------------------------
Total Signals Generated: {stats['signals_generated']}
Signals Per Hour: {stats.get('signals_per_hour', 0):.1f}
Alerts Sent: {stats['alerts_sent']}
Alerts Per Hour: {stats.get('alerts_per_hour', 0):.2f}
Signal Rejection Rate: {rejection_rate:.1f}%

SIGNAL TYPES BREAKDOWN
----------------------
BUY Signals: {stats['signal_types'].get('BUY', 0)} ({stats['signal_types'].get('BUY', 0)/max(stats['signals_generated'], 1)*100:.1f}%)
SELL Signals: {stats['signal_types'].get('SELL', 0)} ({stats['signal_types'].get('SELL', 0)/max(stats['signals_generated'], 1)*100:.1f}%)
NEUTRAL Signals: {stats['signal_types'].get('NEUTRAL', 0)} ({stats['signal_types'].get('NEUTRAL', 0)/max(stats['signals_generated'], 1)*100:.1f}%)

CONFIDENCE ANALYSIS
-------------------
Average Confidence: {stats.get('avg_confidence', 0):.1f}%
Minimum Confidence: {stats.get('min_confidence', 0):.1f}%
Maximum Confidence: {stats.get('max_confidence', 0):.1f}%

MARKET CONDITIONS
-----------------
Price Range: ₹{stats['price_range']['min']:.2f} - ₹{stats['price_range']['max']:.2f}
Price Volatility: {conditions.get('price_volatility', 0):.2f}
Average Candle Range: {conditions.get('avg_range', 0):.2f}
Trend Direction: {conditions.get('trend_direction', 'Unknown').upper()}
Price Momentum: {conditions.get('price_momentum', 0):.2f}%

SIGNAL PATTERNS
---------------
Bullish/Bearish Ratio: {patterns.get('bullish_bearish_ratio', 0):.2f}
Signal Clustering Events: {patterns.get('signal_clustering', 0)}
Signal Transitions: {len(patterns.get('signal_transitions', []))}

REJECTION ANALYSIS
------------------
Total Rejections: {stats['signals_rejected']}
Top Rejection Reasons:
"""
        
        # Add top rejection reasons
        if stats['rejection_reasons']:
            sorted_reasons = sorted(stats['rejection_reasons'].items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_reasons[:5]:
                percentage = (count / stats['signals_rejected']) * 100 if stats['signals_rejected'] > 0 else 0
                report += f"  - {reason}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
SYSTEM HEALTH
-------------
Total Errors: {stats['errors']}
WebSocket Reconnections: {stats['websocket_reconnections']}
Telegram Failures: {stats['telegram_failures']}
Indicators Used: {', '.join(stats['indicators_used'])}

ALERTS SUMMARY
--------------"""
        
        # Add alert details
        for alert in self.alerts:
            report += f"\n  {alert['timestamp'].strftime('%H:%M:%S')} - {alert['type']} Signal at ₹{alert['price']:.2f}"
        
        report += f"""

RECOMMENDATIONS
---------------"""
        
        # Generate recommendations based on analysis
        recommendations = []
        
        if stats.get('signals_per_hour', 0) > 30:
            recommendations.append("⚠️ Excessive signal frequency detected. Consider increasing cooldown period.")
        
        if alert_efficiency < 5:
            recommendations.append("⚠️ Low alert efficiency. Review signal validation criteria.")
        
        if stats.get('avg_confidence', 0) < 40:
            recommendations.append("⚠️ Low average confidence scores. Adjust indicator weights or thresholds.")
        
        if 'Too frequent' in str(stats['rejection_reasons']):
            recommendations.append("⚠️ Signals being rejected for frequency. Implement better rate limiting.")
        
        if conditions.get('price_volatility', 0) < 10:
            recommendations.append("ℹ️ Low market volatility detected. Consider adjusting sensitivity parameters.")
        
        if stats['errors'] > 10:
            recommendations.append("⚠️ High error count detected. Review error logs for system issues.")
        
        for rec in recommendations:
            report += f"\n{rec}"
        
        report += """

================================================================================
                              END OF REPORT
================================================================================
"""
        
        return report
    
    def visualize_performance(self, save_path: str = 'performance_analysis.png'):
        """Create performance visualization charts"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Trading System Performance Analysis', fontsize=16, fontweight='bold')
        
        stats = self.parse_log_file()
        
        # 1. Signal Distribution Pie Chart
        ax1 = axes[0, 0]
        signal_counts = [
            stats['signal_types'].get('BUY', 0),
            stats['signal_types'].get('SELL', 0),
            stats['signal_types'].get('NEUTRAL', 0)
        ]
        colors = ['green', 'red', 'gray']
        ax1.pie(signal_counts, labels=['BUY', 'SELL', 'NEUTRAL'], colors=colors, autopct='%1.1f%%')
        ax1.set_title('Signal Type Distribution')
        
        # 2. Confidence Score Distribution
        ax2 = axes[0, 1]
        if stats['confidence_scores']:
            ax2.hist(stats['confidence_scores'], bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax2.axvline(stats['avg_confidence'], color='red', linestyle='--', label=f'Avg: {stats["avg_confidence"]:.1f}%')
            ax2.set_xlabel('Confidence (%)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Confidence Score Distribution')
            ax2.legend()
        
        # 3. Rejection Reasons Bar Chart
        ax3 = axes[0, 2]
        if stats['rejection_reasons']:
            reasons = list(stats['rejection_reasons'].keys())[:5]
            counts = [stats['rejection_reasons'][r] for r in reasons]
            ax3.barh(range(len(reasons)), counts, color='orange')
            ax3.set_yticks(range(len(reasons)))
            ax3.set_yticklabels([r[:30] + '...' if len(r) > 30 else r for r in reasons])
            ax3.set_xlabel('Count')
            ax3.set_title('Top Rejection Reasons')
        
        # 4. Price Movement Chart
        ax4 = axes[1, 0]
        if self.candles:
            df = pd.DataFrame(self.candles)
            ax4.plot(df['close'], color='blue', linewidth=2)
            ax4.fill_between(range(len(df)), df['low'], df['high'], alpha=0.3, color='gray')
            ax4.set_xlabel('Time (Candles)')
            ax4.set_ylabel('Price (₹)')
            ax4.set_title('Price Movement')
            ax4.grid(True, alpha=0.3)
        
        # 5. Signal Timeline
        ax5 = axes[1, 1]
        if self.signals:
            signal_df = pd.DataFrame(self.signals)
            buy_signals = signal_df[signal_df['type'] == 'BUY']
            sell_signals = signal_df[signal_df['type'] == 'SELL']
            
            if not buy_signals.empty:
                ax5.scatter(range(len(buy_signals)), [1]*len(buy_signals), color='green', s=50, label='BUY')
            if not sell_signals.empty:
                ax5.scatter(range(len(sell_signals)), [0]*len(sell_signals), color='red', s=50, label='SELL')
            
            ax5.set_ylim(-0.5, 1.5)
            ax5.set_xlabel('Signal Index')
            ax5.set_ylabel('Signal Type')
            ax5.set_title('Signal Timeline')
            ax5.legend()
        
        # 6. Performance Metrics Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        metrics_text = f"""
Performance Metrics:
        
Runtime: {stats.get('total_runtime', 'N/A')}
Total Signals: {stats['signals_generated']}
Alerts Sent: {stats['alerts_sent']}
Efficiency: {(stats['alerts_sent']/max(stats['signals_generated'], 1)*100):.1f}%
Avg Confidence: {stats.get('avg_confidence', 0):.1f}%
Error Count: {stats['errors']}
        """
        
        ax6.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center')
        ax6.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path

# Main execution
if __name__ == "__main__":
    # Analyze the provided log file
    analyzer = TradingSystemAnalyzer('trading_system.log')
    
    # Generate performance report
    report = analyzer.generate_performance_report()
    print(report)
    
    # Save report to file
    with open('performance_report.txt', 'w') as f:
        f.write(report)
    
    # Generate visualization
    chart_path = analyzer.visualize_performance()
    print(f"\nPerformance visualization saved to: {chart_path}")
    
    # Generate detailed metrics JSON
    stats = analyzer.parse_log_file()
    patterns = analyzer.analyze_signal_patterns()
    conditions = analyzer.analyze_market_conditions()
    
    detailed_metrics = {
        'statistics': stats,
        'patterns': patterns,
        'market_conditions': conditions,
        'alerts': analyzer.alerts,
        'errors_summary': {
            'total': len(analyzer.errors),
            'recent_5': analyzer.errors[-5:] if analyzer.errors else []
        }
    }
    
    with open('detailed_metrics.json', 'w') as f:
        json.dump(detailed_metrics, f, indent=2, default=str)
    
    print("\nDetailed metrics saved to: detailed_metrics.json")
