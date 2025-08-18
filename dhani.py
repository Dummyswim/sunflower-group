import base64
import json
import logging
import os
import ssl
import sys
import threading
import time
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import websocket
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
# Replace these with your actual base64-encoded credentials
DHAN_ACCESS_TOKEN_B64 = "YOUR_BASE64_DHAN_ACCESS_TOKEN"
DHAN_CLIENT_ID_B64 = "YOUR_BASE64_DHAN_CLIENT_ID"
TELEGRAM_BOT_TOKEN_B64 = "YOUR_BASE64_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"  # e.g., "-123456789"

# ========== LOGGING SETUP ==========
logging.basicConfig(
    filename="nifty_alerts.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ========== TECHNICAL INDICATORS ==========
class TechnicalIndicators:
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        crossover = None
        if len(macd_line) >= 3:
            if (macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]):
                crossover = 'bullish'
            elif (macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]):
                crossover = 'bearish'
        return {
            'macd': round(macd_line.iloc[-1], 4),
            'signal': round(signal_line.iloc[-1], 4),
            'histogram': round(histogram.iloc[-1], 4),
            'crossover': crossover,
            'macd_series': macd_line,
            'signal_series': signal_line
        }

    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        divergence = None
        if len(rsi) >= 10:
            recent_rsi = rsi.tail(10)
            recent_prices = prices.tail(10)
            if (recent_prices.iloc[-1] > recent_prices.iloc[-5] and recent_rsi.iloc[-1] < recent_rsi.iloc[-5]):
                divergence = 'bearish'
            elif (recent_prices.iloc[-1] < recent_prices.iloc[-5] and recent_rsi.iloc[-1] > recent_rsi.iloc[-5]):
                divergence = 'bullish'
        return {'rsi': round(rsi.iloc[-1], 2), 'divergence': divergence}

    @staticmethod
    def calculate_impulse_macd(prices):
        macd_data = TechnicalIndicators.calculate_macd(prices)
        ema_13 = prices.ewm(span=13, adjust=False).mean()
        ema_2 = prices.ewm(span=2, adjust=False).mean()
        current_histogram = macd_data['histogram']
        price_momentum = ema_2.iloc[-1] - ema_13.iloc[-1]
        if current_histogram > 0 and price_momentum > 0:
            impulse_state = 'bullish'
        elif current_histogram < 0 and price_momentum < 0:
            impulse_state = 'bearish'
        else:
            impulse_state = 'neutral'
        return {
            'state': impulse_state,
            'histogram': round(current_histogram, 4),
            'momentum': round(price_momentum, 4)
        }

# ========== DHANHQ WEBSOCKET CLIENT ==========
class DhanWebSocketClient:
    def __init__(self, access_token_b64, client_id_b64):
        self.access_token = base64.b64decode(access_token_b64).decode('utf-8')
        self.client_id = base64.b64decode(client_id_b64).decode('utf-8')
        self.data_buffer = pd.DataFrame()
        self.ws = None
        self.connected = False

    def on_open(self, ws):
        logging.info("WebSocket connection opened.")
        # Subscribe to Nifty50 Index (SecurityId: 11, ExchangeSegment: 1)
        sub_msg = {
            "RequestCode": 11,
            "InstrumentCount": 1,
            "InstrumentList": [{"ExchangeSegment": 1, "SecurityId": "11"}]
        }
        ws.send(json.dumps(sub_msg))
        self.connected = True

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'Touchline' in data:
                touchline = data['Touchline']
                market_data = {
                    'timestamp': datetime.now(),
                    'open': touchline.get('Open', 0),
                    'high': touchline.get('High', 0),
                    'low': touchline.get('Low', 0),
                    'close': touchline.get('Close', 0),
                    'ltp': touchline.get('LastTradedPrice', 0),
                    'volume': touchline.get('TotalTradedQuantity', 0)
                }
                self.add_to_buffer(market_data)
        except Exception as e:
            logging.error(f"Error in on_message: {e}")

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.warning("WebSocket connection closed.")
        self.connected = False

    def add_to_buffer(self, data):
        new_row = pd.DataFrame([data])
        self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)
        if len(self.data_buffer) > 200:
            self.data_buffer = self.data_buffer.tail(200).reset_index(drop=True)

    def connect(self):
        ws_url = "wss://api.dhan.co/v2/websocket/data"
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            header=[
                f"access-token:{self.access_token}",
                f"client-id:{self.client_id}"
            ]
        )
        wst = threading.Thread(target=self.ws.run_forever, kwargs={'sslopt': {"cert_reqs": ssl.CERT_NONE}})
        wst.daemon = True
        wst.start()
        # Wait for connection
        timeout = 10
        while not self.connected and timeout > 0:
            time.sleep(1)
            timeout -= 1
        if not self.connected:
            logging.error("WebSocket connection failed.")
            sys.exit(1)

# ========== TELEGRAM BOT ==========
class TelegramBot:
    def __init__(self, bot_token_b64, chat_id):
        self.bot_token = base64.b64decode(bot_token_b64).decode('utf-8')
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, message, parse_mode="HTML"):
        url = f"{self.base_url}/sendMessage"
        payload = {'chat_id': self.chat_id, 'text': message, 'parse_mode': parse_mode}
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code != 200:
                logging.error(f"Telegram send_message failed: {response.text}")
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Telegram send_message error: {e}")
            return False

    def send_chart(self, message, chart_path):
        url = f"{self.base_url}/sendPhoto"
        try:
            with open(chart_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': self.chat_id, 'caption': message, 'parse_mode': 'HTML'}
                response = requests.post(url, files=files, data=data, timeout=30)
            if response.status_code != 200:
                logging.error(f"Telegram send_chart failed: {response.text}")
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Telegram send_chart error: {e}")
            return False

# ========== ALERT LOGIC ==========
def should_send_alert(macd, rsi, impulse):
    macd_signal = macd.get('crossover', None)
    rsi_value = rsi.get('rsi', 50)
    rsi_ok = 30 <= rsi_value <= 70 and not rsi.get('divergence')
    impulse_state = impulse.get('state', 'neutral')
    impulse_confirmation = impulse_state in ['bullish', 'bearish']
    return macd_signal and rsi_ok and impulse_confirmation

def format_alert_message(price, macd, rsi, impulse, trend, timestamp):
    return f"""🚨 <b>Nifty50 Alert</b> 🚨
Current Price: ₹{price:,.2f}
MACD: {macd['macd']} | Signal: {macd['signal']}
RSI: {rsi['rsi']}
Impulse MACD: {impulse['state'].upper()}
Trend: {trend}
Time: {timestamp.strftime('%H:%M:%S')}
"""

def plot_chart(prices, macd, rsi, filename):
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(3, 1, 1)
    prices.plot(ax=ax1, color='blue', label='Nifty50 Price')
    ax1.set_title('Nifty50 Price')
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    macd['macd_series'].plot(ax=ax2, color='green', label='MACD')
    macd['signal_series'].plot(ax=ax2, color='red', label='Signal')
    ax2.set_title('MACD & Signal')
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3)
    rsi_series = prices.rolling(window=14).apply(lambda x: TechnicalIndicators.calculate_rsi(pd.Series(x))['rsi'])
    rsi_series.plot(ax=ax3, color='purple', label='RSI')
    ax3.axhline(70, color='red', linestyle='--')
    ax3.axhline(30, color='green', linestyle='--')
    ax3.set_title('RSI')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ========== MAIN SCHEDULER ==========
def run_alert_system():
    dhan_client = DhanWebSocketClient(DHAN_ACCESS_TOKEN_B64, DHAN_CLIENT_ID_B64)
    telegram_bot = TelegramBot(TELEGRAM_BOT_TOKEN_B64, TELEGRAM_CHAT_ID)
    dhan_client.connect()
    logging.info("Started Nifty50 alert system.")

    # Wait until 3PM
    while datetime.now().hour < 15:
        logging.info("Waiting for 3PM to start monitoring...")
        time.sleep(60)

    last_alert_time = None
    cooldown_minutes = 15

    while True:
        try:
            if len(dhan_client.data_buffer) < 30:
                logging.info("Not enough data yet, waiting...")
                time.sleep(60)
                continue

            prices = dhan_client.data_buffer['ltp'].astype(float)
            macd = TechnicalIndicators.calculate_macd(prices)
            rsi = TechnicalIndicators.calculate_rsi(prices)
            impulse = TechnicalIndicators.calculate_impulse_macd(prices)
            trend = "Bullish" if macd['crossover'] == 'bullish' else "Bearish" if macd['crossover'] == 'bearish' else "Neutral"

            now = datetime.now()
            if should_send_alert(macd, rsi, impulse):
                if not last_alert_time or (now - last_alert_time).total_seconds() > cooldown_minutes * 60:
                    msg = format_alert_message(prices.iloc[-1], macd, rsi, impulse, trend, now)
                    chart_path = "nifty50_chart.png"
                    plot_chart(prices, macd, rsi, chart_path)
                    telegram_bot.send_chart(msg, chart_path)
                    last_alert_time = now
                    logging.info("Alert sent.")
                else:
                    logging.info("Alert condition met but in cooldown period.")
            else:
                logging.info("No alert condition met.")

            time.sleep(900)  # 15 minutes
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_alert_system()
