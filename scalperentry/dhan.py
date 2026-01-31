"""dhan.py

DhanHQ v2 Live Market Feed (Quote Packet) runner for Nifty Index scalping.

This is the *relay-free* lane:
- Connects to Dhan WS (v2)
- Decodes Quote Packet ticks (resp_code=4) and optional OI packets (resp_code=5)
- Feeds ticks directly to TradeBrain (tradebrain.py)
- Appends TradeBrain decisions to a JSONL file

Single-command run
    export DHAN_ACCESS_TOKEN=...; export DHAN_CLIENT_ID=...
    python dhan.py

Key env vars (new, clean names)
  Required
    DHAN_ACCESS_TOKEN
    DHAN_CLIENT_ID

  Subscription
    EXCHANGE_SEGMENT   (default IDX_I)
    SECURITY_ID        (default 13)

  Logging
    LOG_FILE           (default logs/dhan.log)
    LOG_LEVEL          (default INFO)
    LOG_TO_CONSOLE     (default 0)
    LOG_TICKS          (default 0)
    LOG_EVERY_N        (default 500)

  TradeBrain
    RUN_TRADEBRAIN     (default 1)
    TB_LOG_FILE        (default logs/tradebrain.log)
    TB_JSONL           (default tradebrain_signals.jsonl)
    TB_WRITE_HOLD      (default 0)   # if 1, writes HOLD rows too
    TB_WRITE_ALL       (default 0)   # if 1, writes every closed candle row

Back-compat (old FULL_* names)
- For smooth migration, this script still honors the legacy FULL_* variables as aliases.

Docs reference (packet layout)
- DhanHQ v2 Live Market Feed "Quote Packet" field offsets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple

import websockets

from tradebrain import TradeBrain, TradeBrainConfig, setup_logger


# ----------------------------
# Logging (file-first, IST timestamps)
# ----------------------------

IST_TZ = timezone(timedelta(hours=5, minutes=30))


class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, IST_TZ)
        base = dt.strftime(datefmt) if datefmt else dt.strftime("%Y-%m-%d %H:%M:%S")
        off = dt.strftime("%z")
        off = (off[:3] + ":" + off[3:]) if len(off) == 5 else off
        return f"{base}{off}"


def setup_logging(log_file: str, level: int = logging.INFO, also_console: bool = False) -> logging.Logger:
    """Configure a root logger so websockets + this module land in one file."""
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    fmt = ISTFormatter("%(asctime)s | %(levelname)s | %(message)s")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []

    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    root.addHandler(fh)

    if also_console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        root.addHandler(sh)

    log = logging.getLogger("dhan")
    log.setLevel(level)
    log.propagate = True
    return log


logger = logging.getLogger(__name__)


# ----------------------------
# Config
# ----------------------------

@dataclass
class DhanConfig:
    exchange_segment: str = "IDX_I"
    security_id: int = 13

    ws_ping_interval: int = 30
    ws_ping_timeout: int = 20
    reconnect_delay_base: float = 2.0
    reconnect_delay_cap: float = 60.0
    max_reconnect_attempts: int = 0  # 0 => infinite

    log_every_n: int = 500
    log_ticks: bool = False
    log_tick_summary_sec: int = 600


def _env(key: str, default: str, aliases: Optional[list[str]] = None) -> str:
    v = os.getenv(key)
    if v is not None and str(v).strip() != "":
        return str(v).strip()
    for a in (aliases or []):
        v2 = os.getenv(a)
        if v2 is not None and str(v2).strip() != "":
            return str(v2).strip()
    return default


def _env_int(key: str, default: int, aliases: Optional[list[str]] = None) -> int:
    raw = _env(key, str(default), aliases)
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _env_float(key: str, default: float, aliases: Optional[list[str]] = None) -> float:
    raw = _env(key, str(default), aliases)
    try:
        return float(str(raw).strip())
    except Exception:
        return default


def _env_bool(key: str, default: bool, aliases: Optional[list[str]] = None) -> bool:
    raw = _env(key, "", aliases).strip().lower()
    if raw == "":
        return default
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_log_level(default: int = logging.INFO) -> int:
    raw = _env("LOG_LEVEL", "", aliases=["FULL_LOG_LEVEL"]).strip().upper()
    if not raw:
        return default
    if raw.isdigit():
        try:
            return int(raw)
        except Exception:
            return default
    return int(getattr(logging, raw, default))


def _mask_token(token: str) -> str:
    token = token or ""
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def _build_dhan_ws_url(access_token: str, client_id: str) -> str:
    return "wss://api-feed.dhan.co" + f"?version=2&token={access_token}&clientId={client_id}&authType=2"


def _subscription_payload(cfg: DhanConfig) -> Dict[str, Any]:
    # RequestCode=17 => Subscribe Quote Packet (response_code=4).
    return {
        "RequestCode": 17,
        "InstrumentCount": 1,
        "InstrumentList": [
            {"ExchangeSegment": cfg.exchange_segment, "SecurityId": str(cfg.security_id)}
        ],
    }


# ----------------------------
# Packet decoding (per docs)
# ----------------------------


def _decode_header(data: bytes) -> Tuple[int, int, int, int]:
    """Return (response_code, msg_len, exchange_segment, security_id)."""
    if len(data) < 8:
        raise ValueError("packet too short for response header")
    # Response Header (8 bytes) - Little Endian
    # byte 0: response code (uint8)
    # bytes 1-2: message length (int16)
    # byte 3: exchange segment (uint8)
    # bytes 4-7: security id (int32)
    resp_code, msg_len, exch_seg, sec_id = struct.unpack_from("<BhBi", data, 0)
    return int(resp_code & 0xFF), int(msg_len), int(exch_seg & 0xFF), int(sec_id)


def _epoch_to_iso_utc(epoch: int) -> Optional[str]:
    try:
        if epoch < 946684800 or epoch > 4102444800:  # 2000..2100 sanity
            return None
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
    except Exception:
        return None


def parse_quote_packet(data: bytes) -> Optional[Dict[str, Any]]:
    """Decode Dhan Quote Packet (ResponseCode=4) into a JSON-friendly dict."""
    try:
        resp_code, msg_len, exch_seg, sec_id = _decode_header(data)
    except Exception:
        return None

    if resp_code != 4:
        return None

    if msg_len <= 0 or msg_len > len(data):
        msg_len = len(data)

    payload = data[8:msg_len]
    if len(payload) < 42:
        return None

    try:
        ltp = float(struct.unpack_from("<f", payload, 0)[0])
        ltq = int(struct.unpack_from("<h", payload, 4)[0])
        ltt_epoch = int(struct.unpack_from("<i", payload, 6)[0])
        atp = float(struct.unpack_from("<f", payload, 10)[0])
        volume = int(struct.unpack_from("<i", payload, 14)[0])
        day_open = float(struct.unpack_from("<f", payload, 26)[0])
        day_close = float(struct.unpack_from("<f", payload, 30)[0])
        day_high = float(struct.unpack_from("<f", payload, 34)[0])
        day_low = float(struct.unpack_from("<f", payload, 38)[0])
    except Exception:
        return None

    ts_iso = _epoch_to_iso_utc(ltt_epoch) or datetime.now(timezone.utc).isoformat()

    return {
        "kind": "dhan_quote_packet",
        "response_code": int(resp_code),
        "exchange_segment": int(exch_seg),
        "security_id": int(sec_id),
        "timestamp": ts_iso,
        "ltt_epoch": int(ltt_epoch),
        "ltp": float(ltp),
        "ltq": int(ltq),
        "atp": float(atp),
        "volume": int(volume),
        "day_open": float(day_open),
        "day_close": float(day_close),
        "day_high": float(day_high),
        "day_low": float(day_low),
        "depth": [],
        "spread": 0.0,
        "payload_len": int(len(payload)),
    }


def parse_oi_packet(data: bytes) -> Optional[Dict[str, Any]]:
    """Decode Dhan OI Packet (ResponseCode=5) into a JSON-friendly dict."""
    try:
        resp_code, msg_len, exch_seg, sec_id = _decode_header(data)
    except Exception:
        return None

    if resp_code != 5:
        return None

    if msg_len <= 0 or msg_len > len(data):
        msg_len = len(data)

    payload = data[8:msg_len]
    if len(payload) < 4:
        return None

    try:
        oi = int(struct.unpack_from("<i", payload, 0)[0])
    except Exception:
        return None

    return {
        "kind": "dhan_oi_packet",
        "response_code": int(resp_code),
        "exchange_segment": int(exch_seg),
        "security_id": int(sec_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "oi": int(oi),
        "payload_len": int(len(payload)),
    }


# ----------------------------
# Feed loop
# ----------------------------


async def run_quote_feed(cfg: DhanConfig, tb: Optional[TradeBrain], log: logging.Logger) -> None:
    access_token = _env("DHAN_ACCESS_TOKEN", "", [])
    client_id = _env("DHAN_CLIENT_ID", "", [])
    if not access_token or not client_id:
        raise SystemExit("Missing DHAN_ACCESS_TOKEN or DHAN_CLIENT_ID in environment")

    ws_url = _build_dhan_ws_url(access_token, client_id)
    sub = _subscription_payload(cfg)

    attempt = 0
    tick_count = 0
    total_tick_count = 0
    first_quote = True
    last_summary_monotonic = time.monotonic()
    last_summary_count = 0

    while True:
        attempt += 1
        max_attempts = int(cfg.max_reconnect_attempts)
        if max_attempts > 0 and attempt > max_attempts:
            raise SystemExit(f"Max reconnect attempts reached ({max_attempts})")

        backoff = min(cfg.reconnect_delay_base * (2 ** max(0, attempt - 1)), cfg.reconnect_delay_cap)

        try:
            log.info("[DHAN] connecting to Dhan WS (attempt %d)", attempt)

            async with websockets.connect(
                ws_url,
                ping_interval=cfg.ws_ping_interval,
                ping_timeout=cfg.ws_ping_timeout,
                max_size=None,
                compression=None,
                open_timeout=30,
                close_timeout=10,
            ) as ws:
                log.info(
                    "[DHAN] connected url=wss://api-feed.dhan.co?version=2&token=%s&clientId=%s&authType=2",
                    _mask_token(access_token),
                    _mask_token(client_id),
                )

                await ws.send(json.dumps(sub))
                log.info("[DHAN] subscription sent: %s", sub)

                attempt = 0

                async for msg in ws:
                    if not isinstance(msg, (bytes, bytearray)):
                        log.info("[DHAN] text msg: %s", msg)
                        continue

                    buf = bytes(msg)
                    off = 0
                    now_utc = datetime.now(timezone.utc)
                    recv_ts = now_utc.isoformat()
                    recv_ns = int(time.time_ns())

                    while off + 8 <= len(buf):
                        try:
                            resp_code, msg_len, _, _ = _decode_header(buf[off:off + 8])
                        except Exception:
                            break

                        if msg_len <= 0 or off + msg_len > len(buf):
                            break

                        packet = buf[off:off + msg_len]
                        off += msg_len

                        tick: Optional[Dict[str, Any]] = None
                        if resp_code == 4:
                            tick = parse_quote_packet(packet)
                        elif resp_code == 5:
                            tick = parse_oi_packet(packet)

                        if not tick:
                            continue

                        tick["recv_ts"] = recv_ts
                        tick["recv_ts_ns"] = recv_ns

                        total_tick_count += 1

                        if tick.get("kind") == "dhan_quote_packet":
                            tick_count += 1
                            if first_quote:
                                log.info(
                                    "[DHAN] first quote tick ltp=%.2f ltq=%d vol=%d",
                                    float(tick.get("ltp") or 0.0),
                                    int(tick.get("ltq") or 0),
                                    int(tick.get("volume") or 0),
                                )
                                first_quote = False

                        if cfg.log_ticks:
                            log.info("[DHAN] tick=%s", json.dumps(tick, default=str, separators=(",", ":")))

                        summary_sec = max(0, int(getattr(cfg, "log_tick_summary_sec", 0) or 0))
                        if summary_sec > 0:
                            now_mono = time.monotonic()
                            if now_mono - last_summary_monotonic >= summary_sec:
                                log.info(
                                    "[DHAN] ticks_total=%d ticks_delta=%d interval_sec=%d",
                                    int(total_tick_count),
                                    int(total_tick_count - last_summary_count),
                                    int(summary_sec),
                                )
                                last_summary_monotonic = now_mono
                                last_summary_count = int(total_tick_count)

                        if cfg.log_every_n > 0 and summary_sec <= 0 and (tick_count % cfg.log_every_n == 0) and tick.get("kind") == "dhan_quote_packet":
                            log.info("[DHAN] ticks=%d ltp=%.2f", tick_count, float(tick.get("ltp") or 0.0))

                        if tb is not None:
                            tb.on_tick(tick)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[DHAN] connection error: %s", e)
            log.info("[DHAN] reconnecting in %.1fs", backoff)
            await asyncio.sleep(backoff)


# ----------------------------
# Orchestrator
# ----------------------------


async def run_system() -> None:
    cfg = DhanConfig(
        exchange_segment=_env("EXCHANGE_SEGMENT", "IDX_I", aliases=["FULL_EXCHANGE_SEGMENT"]) or "IDX_I",
        security_id=_env_int("SECURITY_ID", 13, aliases=["FULL_SECURITY_ID"]),
        ws_ping_interval=_env_int("WS_PING_INTERVAL", 30, aliases=["FULL_WS_PING_INTERVAL"]),
        ws_ping_timeout=_env_int("WS_PING_TIMEOUT", 20, aliases=["FULL_WS_PING_TIMEOUT"]),
        reconnect_delay_base=_env_float("RECONNECT_BASE", 2.0, aliases=["FULL_RECONNECT_BASE"]),
        reconnect_delay_cap=_env_float("RECONNECT_CAP", 60.0, aliases=["FULL_RECONNECT_CAP"]),
        max_reconnect_attempts=_env_int("MAX_RECONNECTS", 0, aliases=["FULL_MAX_RECONNECTS"]),
        log_every_n=_env_int("LOG_EVERY_N", 500, aliases=["FULL_LOG_EVERY_N"]),
        log_ticks=_env_bool("LOG_TICKS", False, aliases=["FULL_LOG_TICKS"]),
        log_tick_summary_sec=_env_int("LOG_TICK_SUMMARY_SEC", 600, aliases=["FULL_LOG_TICK_SUMMARY_SEC"]),
    )

    log_file = _env("LOG_FILE", "logs/dhan.log", aliases=["FULL_LOG_FILE"])
    log_level = _parse_log_level(logging.INFO)
    log_to_console = _env_bool("LOG_TO_CONSOLE", False, aliases=["FULL_LOG_TO_CONSOLE"])

    log = setup_logging(log_file=log_file, level=log_level, also_console=log_to_console)

    run_tb = _env_bool("RUN_TRADEBRAIN", True, aliases=["FULL_RUN_TRADEBRAIN"])

    tb: Optional[TradeBrain] = None
    if run_tb:
        tb_cfg = TradeBrainConfig.from_env()  # reads TB_* (and TB_FULL_* aliases)
        tb_log = setup_logger("tradebrain", tb_cfg.log_file, tb_cfg.log_level, tb_cfg.log_to_console)
        tb = TradeBrain(tb_cfg, tb_log)
        log.info("[DHAN] TradeBrain enabled -> %s", tb_cfg.jsonl_path)
    else:
        log.info("[DHAN] TradeBrain disabled (RUN_TRADEBRAIN=0)")

    log.info(
        "[DHAN] config exchange=%s security_id=%s log_every_n=%d log_ticks=%s log_tick_summary_sec=%d log_level=%s",
        cfg.exchange_segment,
        cfg.security_id,
        cfg.log_every_n,
        cfg.log_ticks,
        cfg.log_tick_summary_sec,
        logging.getLevelName(log.level),
    )

    await run_quote_feed(cfg, tb, log)


if __name__ == "__main__":
    try:
        asyncio.run(run_system())
    except KeyboardInterrupt:
        pass
