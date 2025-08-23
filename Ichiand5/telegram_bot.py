"""
Telegram bot module for sending messages and charts with enhanced error handling.
"""
import base64
import requests
import logging
import os
import time
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class TelegramBot:
    """Telegram bot with retry logic and enhanced error handling."""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    MAX_MESSAGE_LENGTH = 4096
    
    def __init__(self, bot_token_b64: str, chat_id: str):
        """
        Initialize Telegram bot with validation.
        
        Args:
            bot_token_b64: Base64 encoded bot token
            chat_id: Telegram chat ID
        """
        try:
            if not bot_token_b64:
                raise ValueError("Bot token cannot be empty")
            if not chat_id:
                raise ValueError("Chat ID cannot be empty")
                
            self.bot_token = base64.b64decode(bot_token_b64).decode("utf-8")
            self.chat_id = chat_id
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
            
            # Validate token format
            if not self.bot_token or ':' not in self.bot_token:
                raise ValueError("Invalid bot token format")
            
            logger.info(f"TelegramBot initialized for chat_id: {chat_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TelegramBot: {e}")
            raise

    def send_message(self, message: str, parse_mode: str = "HTML", retry: bool = True) -> bool:
        """
        Send text message to Telegram with retry logic.
        
        Args:
            message: Message text
            parse_mode: Parse mode (HTML or Markdown)
            retry: Whether to retry on failure
            
        Returns:
            True if successful, False otherwise
        """
        if not message:
            logger.warning("Attempted to send empty message")
            return False
        
        # Truncate message if too long
        if len(message) > self.MAX_MESSAGE_LENGTH:
            logger.warning(f"Message truncated from {len(message)} to {self.MAX_MESSAGE_LENGTH} characters")
            message = message[:self.MAX_MESSAGE_LENGTH-3] + "..."
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        
        retries = self.MAX_RETRIES if retry else 1
        
        for attempt in range(retries):
            try:
                logger.debug(f"Sending message (attempt {attempt + 1}/{retries})")
                response = requests.post(url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    logger.info("Telegram message sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                    
                    # Don't retry on client errors (4xx)
                    if 400 <= response.status_code < 500:
                        logger.error("Client error - not retrying")
                        break
                        
            except requests.exceptions.Timeout:
                logger.error(f"Telegram request timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                logger.error(f"Telegram connection error (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                logger.error(f"Telegram request error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error sending message: {e}")
                break
            
            if attempt < retries - 1:
                time.sleep(self.RETRY_DELAY)
        
        logger.error("Failed to send Telegram message after all retries")
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
        try:
            # Validate inputs
            if not message:
                logger.warning("Chart caption is empty")
                message = "Chart"
            
            if not chart_path:
                logger.error("Chart path is empty")
                return False
            
            chart_path = Path(chart_path)
            
            if not chart_path.exists():
                logger.error(f"Chart file not found: {chart_path}")
                return False
            
            # Check file size (Telegram limit is 10MB for photos)
            file_size = chart_path.stat().st_size
            if file_size > 10 * 1024 * 1024:
                logger.error(f"Chart file too large: {file_size} bytes")
                return False
            
            url = f"{self.base_url}/sendPhoto"
            
            # Truncate caption if needed
            if len(message) > self.MAX_MESSAGE_LENGTH:
                message = message[:self.MAX_MESSAGE_LENGTH-3] + "..."
            
            with open(chart_path, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": message,
                    "parse_mode": "HTML"
                }
                
                logger.debug(f"Sending chart: {chart_path}")
                response = requests.post(url, files=files, data=data, timeout=30)
                
            if response.status_code == 200:
                logger.info("Telegram chart sent successfully")
                
                # Clean up chart file after sending
                try:
                    chart_path.unlink()
                    logger.debug(f"Chart file removed: {chart_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove chart file: {e}")
                
                return True
            else:
                logger.error(f"Telegram send_chart failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram send_chart error: {e}", exc_info=True)
            return False
    
    def test_connection(self) -> bool:
        """Test if the bot can connect to Telegram."""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                bot_info = response.json()
                logger.info(f"Bot connection test successful: {bot_info.get('result', {}).get('username', 'Unknown')}")
                return True
            else:
                logger.error(f"Bot connection test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Bot connection test error: {e}")
            return False
