"""
Telegram bot module for sending trading alerts.
Fixed version without encoding issues.
"""
import base64
import logging
import requests
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TelegramBot:
    """Telegram bot for sending alerts and reports."""
    
    def __init__(self, token_b64: str, chat_id: str):
        """Initialize Telegram bot with encoded token."""
        try:
            self.token = base64.b64decode(token_b64).decode('utf-8')
            self.chat_id = chat_id
            self.base_url = f"https://api.telegram.org/bot{self.token}"
            self.session = requests.Session()
            logger.info("TelegramBot initialized")
        except Exception as e:
            logger.error(f"Telegram bot initialization error: {e}")
            raise
    
    def send_message(self, message: str) -> bool:
        """Send text message to Telegram."""
        try:
            # Clean message of any problematic characters
            clean_message = self._clean_message(message)
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": clean_message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            
            response = self.session.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram send failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send photo with optional caption."""
        try:
            if not Path(photo_path).exists():
                logger.error(f"Photo file not found: {photo_path}")
                return False
            
            clean_caption = self._clean_message(caption)
            
            url = f"{self.base_url}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'caption': clean_caption,
                    'parse_mode': 'HTML'
                }
                
                response = self.session.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                logger.debug("Telegram photo sent successfully")
                return True
            else:
                logger.error(f"Telegram photo send failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram photo send error: {e}")
            return False
    
    def _clean_message(self, message: str) -> str:
        """Clean message of problematic characters."""
        # Replace common emojis with text equivalents
        replacements = {
            '🚀': '[LAUNCH]',
            '📊': '[CHART]',
            '✅': '[OK]',
            '❌': '[X]',
            '⚠️': '[WARNING]',
            '📈': '[UP]',
            '📉': '[DOWN]',
            '💪': '[STRONG]',
            '⏱️': '[TIMER]',
            '🎯': '[TARGET]',
            '🟢': '[GREEN]',
            '🔴': '[RED]',
            '⚪': '[WHITE]',
            '═': '=',
            '₹': 'Rs',
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v'
        }
        
        clean = message
        for old, new in replacements.items():
            clean = clean.replace(old, new)
        
        # Remove any remaining non-ASCII characters
        clean = ''.join(char if ord(char) < 128 else '' for char in clean)
        
        return clean
