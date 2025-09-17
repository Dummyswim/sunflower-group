"""
Enhanced Telegram bot module with connection pooling and retry mechanism.
Fixes RemoteDisconnected errors.
"""
import base64
import logging
import requests
from typing import Optional
from pathlib import Path
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from http.client import RemoteDisconnected
from requests.exceptions import ChunkedEncodingError, SSLError


logger = logging.getLogger(__name__)

class TelegramBot:
    """Telegram bot with enhanced connection handling and retry logic."""
    
    def __init__(self, token_b64: str, chat_id: str):
        """Initialize Telegram bot with robust connection handling."""
        try:
            self.token = base64.b64decode(token_b64).decode('utf-8')
            self.chat_id = chat_id
            self.base_url = f"https://api.telegram.org/bot{self.token}"
            
            # Create session with connection pooling and retry strategy
            self.session = self._create_robust_session()
            
            logger.info("TelegramBot initialized with enhanced connection handling")
            logger.debug(f"Chat ID: {chat_id}, Base URL configured")
        except Exception as e:
            logger.error(f"Telegram bot initialization error: {e}")
            raise

    def _create_robust_session(self) -> requests.Session:
        """Create a session with retry strategy and connection pooling."""
        logger.debug("Creating robust session with retry strategy")
        
        session = requests.Session()
        
        # Configure retry strategy - FIXED for urllib3 2.0+
        # retry_strategy = Retry(
        #     total=5,  # Increased retries
        #     status_forcelist=[429, 500, 502, 503, 504, 104],
        #     allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],  # Changed from method_whitelist
        #     backoff_factor=2,
        #     respect_retry_after_header=True  # Added this for better rate limit handling
        # )
        
        retry_strategy = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=1.5,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            raise_on_status=False
        )

        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,  # Increased
            pool_maxsize=30,      # Increased
            pool_block=True       # Block when pool is full
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers to keep connection alive
        session.headers.update({
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=30, max=100'
        })
        
        logger.info("Session configured with 5 retries, pool 20-30, keep-alive enabled")
        return session

    def send_message(self, message: str, retry_count: int = 3) -> bool:
        """Send text message to Telegram with retry mechanism."""
        logger.debug(f"Attempting to send message (length: {len(message)} chars)")
        
        for attempt in range(retry_count):
            try:
                # Clean message of any problematic characters
                clean_message = self._clean_message(message)
                logger.debug(f"Message cleaned, attempt {attempt + 1}/{retry_count}")
                

                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": clean_message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                }
                timeout = (5, 15)
                self._log_http_attempt("sendMessage", url, attempt + 1, timeout)
                resp = self.session.post(url, json=payload, timeout=timeout)
                self._log_http_response("sendMessage", resp)
                if resp.status_code == 200:
                    logger.info(f"Telegram message sent successfully on attempt {attempt + 1}")
                    return True
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get('Retry-After', 5))
                    logger.warning(f"[TG] Rate limited (429). Backing off {retry_after}s")
                    time.sleep(retry_after)
                    continue
                logger.warning(f"[TG] Non-200 from Telegram: {resp.status_code} {resp.text[:140]}")
                            

                        
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    # Recreate session if connection errors persist
                    if attempt == 1:
                        logger.info("Recreating session due to persistent connection errors")
                        self.session.close()
                        self.session = self._create_robust_session()
                        
            except requests.exceptions.Timeout as e:
                logger.error(f"Request timeout on attempt {attempt + 1}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                et = type(e).__name__
                logger.warning(f"[TG] sendMessage exception on attempt {attempt + 1}: {et} → {e}")
                if attempt < retry_count - 1 and self._is_transient_exc(e):
                    wait_time = 2 ** attempt
                    logger.info(f"[TG] Transient error; retrying in {wait_time}s")
                    time.sleep(wait_time)
                    if attempt == 1:
                        logger.info("[TG] Recreating session after repeated transient errors")
                        try:
                            self.session.close()
                        except Exception:
                            pass
                        self.session = self._create_robust_session()
                    continue
                if attempt < retry_count - 1:
                    time.sleep(1)


        
        logger.error(f"Failed to send message after {retry_count} attempts")
        return False


    def _log_http_attempt(self, action: str, url: str, attempt: int, timeout: tuple):
        logger.info(f"[TG] {action} attempt {attempt} → url={url}, timeout={timeout}")

    def _log_http_response(self, action: str, resp: requests.Response):
        try:
            logger.info(f"[TG] {action} response → status={resp.status_code}, ok={resp.ok}, len={len(resp.content)}")
        except Exception:
            logger.info(f"[TG] {action} response → status={getattr(resp,'status_code',None)}, ok={getattr(resp,'ok',None)}")

    def _is_transient_exc(self, e: Exception) -> bool:
        return isinstance(e, (RemoteDisconnected, ChunkedEncodingError, SSLError, requests.exceptions.ConnectionError))


    
    def send_photo(self, photo_path: str, caption: str = "", retry_count: int = 3) -> bool:
        """Send photo with optional caption and retry mechanism."""
        logger.debug(f"Attempting to send photo: {photo_path}")
        
        if not Path(photo_path).exists():
            logger.error(f"Photo file not found: {photo_path}")
            return False
        
        for attempt in range(retry_count):
            try:
                clean_caption = self._clean_message(caption)
                
                
                url = f"{self.base_url}/sendPhoto"
                timeout = (10, 60)
                self._log_http_attempt("sendPhoto", url, attempt + 1, timeout)
                with open(photo_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {
                        'chat_id': self.chat_id,
                        'caption': clean_caption,
                        'parse_mode': 'HTML'
                    }
                    resp = self.session.post(url, files=files, data=data, timeout=timeout)
                self._log_http_response("sendPhoto", resp)
                if resp.status_code == 200:
                    logger.info(f"Telegram photo sent successfully on attempt {attempt + 1}")
                    return True
                logger.warning(f"[TG] sendPhoto non-200 → status={resp.status_code}, body={resp.text[:140]}")
                logger.warning(f"Photo upload failed with status {resp.status_code}")
                    
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error uploading photo on attempt {attempt + 1}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    if attempt == 1:
                        self.session.close()
                        self.session = self._create_robust_session()
                        
            except Exception as e:
                et = type(e).__name__
                logger.warning(f"[TG] sendPhoto exception on attempt {attempt + 1}: {et} → {e}")
                if attempt < retry_count - 1 and self._is_transient_exc(e):
                    wait_time = 2 ** attempt
                    logger.info(f"[TG] Transient error; retrying in {wait_time}s")
                    time.sleep(wait_time)
                    if attempt == 1:
                        logger.info("[TG] Recreating session after repeated transient errors (photo)")
                        try:
                            self.session.close()
                        except Exception:
                            pass
                        self.session = self._create_robust_session()
                    continue
                if attempt < retry_count - 1:
                    time.sleep(2)

        
        logger.error(f"Failed to send photo after {retry_count} attempts")
        return False
    
    def _clean_message(self, message: str) -> str:
        """Clean message of problematic characters."""
        logger.debug("Cleaning message of special characters")
        
        # Remove all emojis and special characters
        clean = message
        
        # Remove any non-ASCII characters
        clean = ''.join(char if ord(char) < 128 else '' for char in clean)
        
        # Ensure message isn't empty after cleaning
        if not clean.strip():
            clean = "Message contained only special characters"
            
        logger.debug(f"Message cleaned: original length {len(message)}, cleaned length {len(clean)}")
        return clean

    def __del__(self):
        """Clean up session on deletion."""
        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
                logger.debug("Telegram bot session closed")
        except Exception as e:
            logger.debug(f"Session cleanup error (ignored): {e}")
