"""
Telegram bot module for sending messages and charts.
"""
import base64
import requests
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, bot_token_b64: str, chat_id: str):
        """
        Initialize Telegram bot.
        
        Args:
            bot_token_b64: Base64 encoded bot token
            chat_id: Telegram chat ID
        """
        self.bot_token = base64.b64decode(bot_token_b64).decode("utf-8")
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        logger.info(f"TelegramBot initialized for chat_id: {chat_id}")

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send text message to Telegram.
        
        Args:
            message: Message text
            parse_mode: Parse mode (HTML or Markdown)
            
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram send_message failed: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram send_message error: {e}")
            return False

    def send_chart(self, message: str, chart_path: str) -> bool:
        """
        Send chart image with caption to Telegram.
        
        Args:
            message: Caption for the chart
            chart_path: Path to the chart image file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(chart_path):
            logger.error(f"Chart file not found: {chart_path}")
            return False
            
        url = f"{self.base_url}/sendPhoto"
        
        try:
            with open(chart_path, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": message,
                    "parse_mode": "HTML"
                }
                response = requests.post(url, files=files, data=data, timeout=30)
                
            if response.status_code == 200:
                logger.info("Telegram chart sent successfully")
                # Clean up chart file after sending
                try:
                    os.remove(chart_path)
                    logger.debug(f"Chart file removed: {chart_path}")
                except OSError:
                    pass
                return True
            else:
                logger.error(f"Telegram send_chart failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram send_chart error: {e}")
            return False