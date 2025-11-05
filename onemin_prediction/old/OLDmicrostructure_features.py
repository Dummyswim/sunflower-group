"""
Microstructure feature extraction for AR-NMS.
"""
import numpy as np
import pandas as pd
from collections import deque

class MicrostructureExtractor:
    def __init__(self, max_history=1000):
        self.ticks = deque(maxlen=max_history)
        self.order_books = deque(maxlen=max_history)
        self.quote_lifespans = deque(maxlen=max_history)
        self.last_quote_times = {}

    def update(self, tick, order_book, timestamp):
        self.ticks.append(tick)
        self.order_books.append(order_book)
        # Track quote lifespan
        for quote_id, quote in order_book.get('quotes', {}).items():
            if quote_id not in self.last_quote_times:
                self.last_quote_times[quote_id] = timestamp
            elif quote.get('cancelled'):
                lifespan = timestamp - self.last_quote_times[quote_id]
                self.quote_lifespans.append(lifespan)
                del self.last_quote_times[quote_id]

    # REMOVED: trade_imbalance_velocity used buy_volume/sell_volume (volume-dependent) [[16]]
    # def trade_imbalance_velocity(self, window=5):
    #     df = pd.DataFrame(list(self.ticks)[-window:])
    #     return (df['buy_volume'].sum() - df['sell_volume'].sum()) / window if not df.empty else 0

    # CHANGED: volume-free alternative using price uptick/downtick counts
    def price_imbalance_velocity(self, window=5):  # NEW: replaces trade_imbalance_velocity
        df = pd.DataFrame(list(self.ticks)[-window:])
        if df.empty or 'ltp' not in df:
            return 0.0
        try:
            d = np.diff(df['ltp'].astype(float).values)
            d = d[~np.isclose(d, 0.0)]
            ups = np.sum(d > 0)
            dns = np.sum(d < 0)
            tot = ups + dns
            return float((ups - dns) / max(1, tot))
        except Exception:
            return 0.0

    # REMOVED: liquidity_absorption_rate used executed_volume/available_liquidity (volume-dependent) [[16]]
    # def liquidity_absorption_rate(self, window=10):
    #     df = pd.DataFrame(list(self.ticks)[-window:])
    #     return df['executed_volume'].sum() / (df['available_liquidity'].sum() + 1e-9) if not df.empty else 0

    # CHANGED: volume-free micro-range tightness (higher means tighter range)
    def price_range_tightness(self, window=10):  # NEW: replaces liquidity_absorption_rate
        df = pd.DataFrame(list(self.ticks)[-window:])
        if df.empty or 'ltp' not in df:
            return 0.0
        try:
            px = df['ltp'].astype(float)
            rng = float(px.max() - px.min())
            last = float(px.iloc[-1])
            base = max(1e-9, abs(last))
            tightness = 1.0 - min(1.0, rng / base)  # 0..1 (1 = tight)
            return float(tightness)
        except Exception:
            return 0.0

    def quote_lifespan(self):
        return np.mean(self.quote_lifespans) if self.quote_lifespans else 0

    def market_entropy(self, window=30):
        df = pd.DataFrame(list(self.ticks)[-window:])
        if df.empty or 'ltp' not in df:
            return 0.0  # CHANGED: NaN-safe return; previously assumed df not empty
        directions = np.sign(df['ltp'].astype(float).diff().fillna(0))
        p_up = (directions > 0).mean()
        p_down = (directions < 0).mean()
        entropy = 0.0
        # CHANGED: guard against None or NaN probabilities
        for p in [p_up, p_down]:
            if p and p > 0:
                entropy -= p * np.log2(p)
        return float(entropy)
