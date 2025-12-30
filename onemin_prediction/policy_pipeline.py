#!/usr/bin/env python3
"""
Policy pipeline for rule-as-teacher: predicts success probability for BUY/SELL.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

logger = logging.getLogger(__name__)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _logit_clip(p: float) -> float:
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def _load_schema_cols(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(data, dict):
        cols = data.get("columns") or data.get("feature_names") or data.get("features")
        if isinstance(cols, list) and all(isinstance(x, str) for x in cols) and cols:
            return list(cols)
        if "columns" in data and isinstance(data["columns"], dict):
            return list(data["columns"].keys())

    if isinstance(data, list) and all(isinstance(x, str) for x in data) and data:
        return list(data)

    return None


def _load_calibrator(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    try:
        a = float(data.get("a"))
        b = float(data.get("b"))
        if not np.isfinite(a) or not np.isfinite(b):
            return None
    except Exception:
        return None
    return {
        "a": a,
        "b": b,
        "n": int(data.get("n", 0)),
        "auc": float(data.get("auc", 0.0)),
        "inverted": bool(data.get("inverted", False)),
    }


@dataclass
class PolicyModel:
    booster: Any
    name: str
    feature_names: Optional[List[str]] = None
    num_features: Optional[int] = None

    def predict_success(self, X: np.ndarray) -> float:
        if xgb is None:
            raise RuntimeError("xgboost not available")
        dm = xgb.DMatrix(X)
        p = self.booster.predict(dm)
        p = np.asarray(p, dtype=float)
        if p.ndim == 0:
            return float(np.clip(p, 1e-9, 1 - 1e-9))
        if p.ndim >= 1:
            return float(np.clip(p.ravel()[0], 1e-9, 1 - 1e-9))
        return 0.5


class PolicyPipeline:
    def __init__(
        self,
        buy_model: PolicyModel,
        sell_model: PolicyModel,
        move_model: Optional[PolicyModel] = None,
        *,
        schema_cols: List[str],
        calib_buy_path: Optional[str] = None,
        calib_sell_path: Optional[str] = None,
    ) -> None:
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.move_model = move_model
        self.feature_schema_names = list(schema_cols)
        self._calib_buy_path = calib_buy_path or ""
        self._calib_sell_path = calib_sell_path or ""
        self._calib_buy = None
        self._calib_sell = None
        self.last_p_success_raw: Optional[float] = None
        self.last_p_success_calib: Optional[float] = None
        self.last_teacher_dir: Optional[str] = None
        self.last_p_move_raw: Optional[float] = None

    def _align_features(
        self,
        feature_names: Optional[List[str]],
        feature_values: List[float],
        *,
        model: PolicyModel,
    ) -> np.ndarray:
        if not self.feature_schema_names:
            raise RuntimeError("feature schema is empty")

        if not feature_names or len(feature_names) != len(feature_values):
            raise RuntimeError("feature names mismatch")

        incoming = {str(k): _safe_float(v, 0.0) for k, v in zip(feature_names, feature_values)}
        if model.feature_names:
            if model.num_features is not None and len(model.feature_names) != int(model.num_features):
                logger.warning(
                    "[POLICY] model %s feature_names=%d num_features=%s mismatch; aligning by feature_names",
                    model.name,
                    len(model.feature_names),
                    str(model.num_features),
                )
            aligned = [incoming.get(k, 0.0) for k in model.feature_names]
        else:
            aligned = [incoming.get(k, 0.0) for k in self.feature_schema_names]
            if model.num_features is not None:
                if len(aligned) > model.num_features:
                    logger.warning(
                        "[POLICY] model %s expects %s features; truncating from %d schema cols",
                        model.name,
                        str(model.num_features),
                        len(aligned),
                    )
                    aligned = aligned[: int(model.num_features)]
                elif len(aligned) < model.num_features:
                    logger.warning(
                        "[POLICY] model %s expects %s features; padding from %d schema cols",
                        model.name,
                        str(model.num_features),
                        len(aligned),
                    )
                    aligned.extend([0.0] * (int(model.num_features) - len(aligned)))
        X = np.asarray(aligned, dtype=float).reshape(1, -1)
        if np.isnan(X).any() or np.isinf(X).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _apply_calibration(self, p_raw: float, calib: Optional[Dict[str, Any]]) -> float:
        if calib is None:
            return float(np.clip(p_raw, 1e-9, 1 - 1e-9))
        if bool(calib.get("inverted", False)):
            p_raw = 1.0 - float(p_raw)
        a = float(calib.get("a", 1.0))
        b = float(calib.get("b", 0.0))
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
            return float(np.clip(p_raw, 1e-9, 1 - 1e-9))
        z = a * _logit_clip(p_raw) + b
        return _sigmoid(z)

    def compute_indicator_score(self, feats: Dict[str, Any]) -> float:
        weights = {
            'ema_trend': 0.35,
            'micro_slope': 0.25,
            'imbalance': 0.20,
            'mean_drift': 0.20,
        }
        score = 0.0
        for k, w in weights.items():
            try:
                v = float(feats.get(k, 0.0))
            except Exception:
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            v = float(np.clip(v, -1.0, 1.0))
            score += w * v
        return float(np.clip(score, -1.0, 1.0))

    def reload_calibration(self) -> None:
        if self._calib_buy_path:
            self._calib_buy = _load_calibrator(self._calib_buy_path)
        if self._calib_sell_path:
            self._calib_sell = _load_calibrator(self._calib_sell_path)

    def replace_models(self, *, buy_model: Any = None, sell_model: Any = None) -> None:
        def _wrap(model: Any, name: str) -> Optional[PolicyModel]:
            if model is None:
                return None
            if isinstance(model, PolicyModel):
                return model
            booster = None
            if hasattr(model, "get_booster"):
                try:
                    booster = model.get_booster()
                except Exception:
                    booster = None
            if booster is None and hasattr(model, "booster"):
                booster = getattr(model, "booster", None)
            if booster is None and hasattr(model, "predict"):
                booster = model
            if booster is None:
                return None
            feature_names = None
            num_features = None
            try:
                feature_names = list(getattr(booster, "feature_names", None) or []) or None
            except Exception:
                feature_names = None
            try:
                num_features = int(booster.num_features())
            except Exception:
                num_features = None
            return PolicyModel(booster=booster, name=name, feature_names=feature_names, num_features=num_features)

        updated = False
        bm = _wrap(buy_model, "policy_buy")
        if bm is not None:
            self.buy_model = bm
            updated = True
        sm = _wrap(sell_model, "policy_sell")
        if sm is not None:
            self.sell_model = sm
            updated = True
        if updated:
            self.reload_calibration()

    def predict_success(
        self,
        *,
        feature_names: List[str],
        feature_values: List[float],
        teacher_dir: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        teacher_dir = str(teacher_dir or "").upper()
        if teacher_dir not in ("BUY", "SELL"):
            self.last_teacher_dir = teacher_dir
            self.last_p_success_raw = None
            self.last_p_success_calib = None
            return None, None

        model = self.buy_model if teacher_dir == "BUY" else self.sell_model
        X = self._align_features(feature_names, feature_values, model=model)
        p_raw = model.predict_success(X)
        calib = self._calib_buy if teacher_dir == "BUY" else self._calib_sell
        p_cal = self._apply_calibration(p_raw, calib)

        self.last_teacher_dir = teacher_dir
        self.last_p_success_raw = float(p_raw)
        self.last_p_success_calib = float(p_cal)
        return float(p_raw), float(p_cal)

    def predict_move(
        self,
        *,
        feature_names: List[str],
        feature_values: List[float],
    ) -> Optional[float]:
        if self.move_model is None:
            self.last_p_move_raw = None
            return None
        model = self.move_model
        X = self._align_features(feature_names, feature_values, model=model)
        p_raw = model.predict_success(X)
        self.last_p_move_raw = float(p_raw)
        return float(p_raw)


def load_policy_models(
    *,
    buy_path: str,
    sell_path: str,
    schema_path: Optional[str] = None,
    calib_buy_path: Optional[str] = None,
    calib_sell_path: Optional[str] = None,
    move_path: Optional[str] = None,
) -> PolicyPipeline:
    if xgb is None:
        raise RuntimeError("xgboost is required for policy models")

    buy_path = str(buy_path or "").strip()
    sell_path = str(sell_path or "").strip()
    if not buy_path or not Path(buy_path).exists():
        raise FileNotFoundError(f"BUY policy model not found: {buy_path}")
    if not sell_path or not Path(sell_path).exists():
        raise FileNotFoundError(f"SELL policy model not found: {sell_path}")

    booster_buy = xgb.Booster()
    booster_buy.load_model(buy_path)
    booster_sell = xgb.Booster()
    booster_sell.load_model(sell_path)
    try:
        buy_feat_names = list(getattr(booster_buy, "feature_names", None) or []) or None
    except Exception:
        buy_feat_names = None
    try:
        sell_feat_names = list(getattr(booster_sell, "feature_names", None) or []) or None
    except Exception:
        sell_feat_names = None
    try:
        buy_num = int(booster_buy.num_features())
    except Exception:
        buy_num = None
    try:
        sell_num = int(booster_sell.num_features())
    except Exception:
        sell_num = None

    if not schema_path:
        schema_path = os.getenv("POLICY_SCHEMA_COLS_PATH", "").strip() or os.getenv("FEATURE_SCHEMA_COLS_PATH", "").strip()

    schema_cols = None
    if schema_path:
        schema_cols = _load_schema_cols(Path(schema_path))
    if not schema_cols:
        raise RuntimeError("POLICY_SCHEMA_COLS_PATH is required for policy models")

    move_model = None
    move_path = str(move_path or "").strip() if move_path is not None else ""
    if not move_path:
        move_path = os.getenv("POLICY_MOVE_PATH", "").strip()
    if move_path:
        if not Path(move_path).exists():
            logger.warning("[POLICY] move model not found: %s (move head disabled)", move_path)
        else:
            booster_move = xgb.Booster()
            booster_move.load_model(move_path)
            try:
                move_feat_names = list(getattr(booster_move, "feature_names", None) or []) or None
            except Exception:
                move_feat_names = None
            try:
                move_num = int(booster_move.num_features())
            except Exception:
                move_num = None
            move_model = PolicyModel(
                booster=booster_move,
                name="policy_move",
                feature_names=move_feat_names,
                num_features=move_num,
            )

    pipe = PolicyPipeline(
        buy_model=PolicyModel(booster=booster_buy, name="policy_buy", feature_names=buy_feat_names, num_features=buy_num),
        sell_model=PolicyModel(booster=booster_sell, name="policy_sell", feature_names=sell_feat_names, num_features=sell_num),
        move_model=move_model,
        schema_cols=schema_cols,
        calib_buy_path=calib_buy_path,
        calib_sell_path=calib_sell_path,
    )
    pipe.reload_calibration()
    logger.info(
        "[POLICY] loaded BUY=%s SELL=%s MOVE=%s schema_n=%d",
        buy_path,
        sell_path,
        move_path or "<none>",
        len(schema_cols),
    )
    return pipe
