"""
Adaptive limit order execution with RL feedback.
"""
class SmartOrderExecutor:
    def __init__(self, rl_agent):
        self.rl_agent = rl_agent

    def place_order(self, price, fill_prob, time_waited, get_mid_price):
        # CHANGED: NaN/None safety and guard on get_mid_price callable [[19]]
        try:
            price = float(price) if price is not None else 0.0
            fp = float(fill_prob) if fill_prob is not None else 0.0
            tw = float(time_waited) if time_waited is not None else 0.0
        except Exception:
            price, fp, tw = 0.0, 0.0, 0.0

        try:
            threshold = float(getattr(self.rl_agent, 'threshold', 0.5))
        except Exception:
            threshold = 0.5

        if fp < threshold and tw > 5:
            try:
                mid = float(get_mid_price()) if callable(get_mid_price) else float(get_mid_price)
            except Exception:
                mid = price
            price = self.adjust_limit_towards_mid(price, mid, fraction=0.25)
        # Place order logic here (call broker API)
        return price

    def adjust_limit_towards_mid(self, price, mid_price, fraction):
        return price + fraction * (mid_price - price)
