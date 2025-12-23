# model_bundle.py
"""
Model bundle utilities: versioned policy artifacts + atomic promotion via symlink.

Goal:
- Never serve a model whose schema doesn't match.
- Never partially write schema/model/calibrator into "production".
- Promote a complete bundle atomically by switching a symlink.

Layout:
trained_models/
  bundles/
    20251212_144857/
      policy_buy.json
      policy_sell.json
      calib_buy.json
      calib_sell.json
      policy_schema_cols.json
      policy_schema_cols.txt
      feature_schema_cols.json
      feature_schema_cols.txt
      manifest.json
  production  -> bundles/20251212_144857  (symlink)

Env:
- MODEL_BUNDLES_DIR (default: trained_models/bundles)
- MODEL_PRODUCTION_LINK (default: trained_models/production)
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


def compute_schema_hash(cols: List[str]) -> str:
    h = hashlib.sha1()
    h.update(",".join(cols).encode("utf-8"))
    return h.hexdigest()


def load_schema_cols(schema_path: Path) -> List[str]:
    if not schema_path.exists():
        raise FileNotFoundError(f"schema file not found: {schema_path}")
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        cols = data.get("columns") or data.get("feature_names") or data.get("features")
        if isinstance(cols, list) and all(isinstance(x, str) for x in cols):
            return cols
        # dict mapping feat->idx
        if "columns" in data and isinstance(data["columns"], dict):
            return list(data["columns"].keys())
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    raise ValueError(f"Unrecognized schema format: {schema_path}")


@dataclass
class BundlePaths:
    bundle_dir: Path
    policy_buy_path: Path
    policy_sell_path: Path
    calib_buy_path: Path
    calib_sell_path: Path
    schema_cols_json: Path
    schema_cols_txt: Path
    base_schema_cols_json: Path
    base_schema_cols_txt: Path
    manifest_path: Path


def make_bundle_dir(
    bundles_dir: Path,
    ts_tag: Optional[str] = None,
) -> Path:
    bundles_dir.mkdir(parents=True, exist_ok=True)
    if ts_tag is None:
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
    bundle_dir = bundles_dir / ts_tag
    # If collision, add suffix
    if bundle_dir.exists():
        i = 1
        while (bundles_dir / f"{ts_tag}_{i}").exists():
            i += 1
        bundle_dir = bundles_dir / f"{ts_tag}_{i}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir


def bundle_paths(bundle_dir: Path) -> BundlePaths:
    return BundlePaths(
        bundle_dir=bundle_dir,
        policy_buy_path=bundle_dir / "policy_buy.json",
        policy_sell_path=bundle_dir / "policy_sell.json",
        calib_buy_path=bundle_dir / "calib_buy.json",
        calib_sell_path=bundle_dir / "calib_sell.json",
        schema_cols_json=bundle_dir / "policy_schema_cols.json",
        schema_cols_txt=bundle_dir / "policy_schema_cols.txt",
        base_schema_cols_json=bundle_dir / "feature_schema_cols.json",
        base_schema_cols_txt=bundle_dir / "feature_schema_cols.txt",
        manifest_path=bundle_dir / "manifest.json",
    )


def validate_bundle_feature_counts(
    policy_buy_path: Path,
    policy_sell_path: Path,
    schema_cols: List[str],
) -> Tuple[bool, str]:
    """
    Returns (ok, reason). Does not raise unless file is missing.
    """
    import xgboost as xgb  # local import

    if not policy_buy_path.exists():
        return False, f"missing policy buy model: {policy_buy_path}"
    if not policy_sell_path.exists():
        return False, f"missing policy sell model: {policy_sell_path}"

    booster_buy = xgb.Booster()
    booster_buy.load_model(str(policy_buy_path))
    n_buy = int(booster_buy.num_features())

    booster_sell = xgb.Booster()
    booster_sell.load_model(str(policy_sell_path))
    n_sell = int(booster_sell.num_features())

    n_schema = int(len(schema_cols))
    if n_buy != n_schema:
        return False, f"policy_buy expects {n_buy} but schema has {n_schema}"
    if n_sell != n_schema:
        return False, f"policy_sell expects {n_sell} but schema has {n_schema}"

    return True, "ok"


def write_manifest(
    manifest_path: Path,
    *,
    schema_cols: List[str],
    notes: Dict[str, Any],
    data_range: Optional[Dict[str, Any]] = None,
    source_counts: Optional[Dict[str, Any]] = None,
    validation_report: Optional[Dict[str, Any]] = None,
) -> None:
    manifest = {
        "schema_cols_n": len(schema_cols),
        "schema_hash": compute_schema_hash(schema_cols),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_range": data_range or {},
        "source_counts": source_counts or {},
        "validation_report": validation_report or {},
        "notes": notes or {},
    }
    atomic_write_json(manifest_path, manifest)


def promote_bundle_symlink(
    bundle_dir: Path,
    production_link: Path,
) -> None:
    """
    Atomically repoint 'production_link' symlink to 'bundle_dir'.
    If production_link is an existing directory (not a symlink), we refuse,
    because that's not atomic-safe.
    """
    production_link.parent.mkdir(parents=True, exist_ok=True)

    if production_link.exists() and not production_link.is_symlink():
        raise RuntimeError(
            f"{production_link} exists and is not a symlink. "
            f"Move it aside and create a symlink instead."
        )

    tmp_link = production_link.with_name(production_link.name + ".tmp")
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        # Create symlink to bundle_dir (use relative to keep it portable)
        rel_target = os.path.relpath(bundle_dir, start=production_link.parent)
        tmp_link.symlink_to(rel_target)
        os.replace(tmp_link, production_link)
    finally:
        try:
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink()
        except Exception:
            pass


def resolve_production_bundle() -> Path:
    """
    Returns the resolved active bundle directory for production link.
    """
    prod = Path(os.getenv("MODEL_PRODUCTION_LINK", "trained_models/production"))
    if not prod.exists():
        raise FileNotFoundError(f"production link not found: {prod}")
    return prod.resolve()
