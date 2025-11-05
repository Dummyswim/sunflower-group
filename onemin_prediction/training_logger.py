import csv
import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TrainingLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ts","decision","label","buy_prob","alpha","tradeable","is_flat","tick_count","features","latent"])
            logger.info(f"[TRAIN] Created log file: {path}")

    def append(self, ts: str, decision: str, label: str, buy_prob: float, alpha: float,
               tradeable: bool, is_flat: bool, tick_count: int,
               features: Dict[str, float], latent: Optional[List[float]]):
        try:
            feat_str = ";".join(f"{k}={float(v):.8f}" for k, v in features.items())
            lat_str = "" if latent is None else "|".join(f"{float(x):.8f}" for x in latent)
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([ts, decision, label, f"{buy_prob:.6f}", f"{alpha:.6f}",
                                        str(tradeable), str(is_flat), tick_count, feat_str, lat_str])
            logger.debug(f"[TRAIN] Append ok: {ts} label={label}")
        except Exception as e:
            logger.warning(f"[TRAIN] Append failed: {e}")
