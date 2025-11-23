#!/usr/bin/env python3
import os
import json
import sys
import numpy as np
import pandas as pd

def main():
    # Inputs
    feature_log_path = os.getenv("FEATURE_LOG", "trained_models/production/feature_log.csv")
    xgb_out_path = os.getenv(
        "XGB_PATH",
        "/home/hanumanth/Documents/sunflower-group_2/onemin_prediction/trained_models/production/xgb.json"
    )

    # 1) Parse feature_log using your existing parser
    from online_trainer import _parse_feature_csv, _build_datasets, _train_xgb

    df = _parse_feature_csv(feature_log_path, min_rows=150)  # allow smaller set for this one-shot
    if df is None or df.empty:
        print(f"[RETRAIN] Not enough rows in {feature_log_path}. Collect more data first.")
        sys.exit(2)

    # 2) Build datasets (uses ALL directionals BUY/SELL)
    dir_ds, _, feat_cols = _build_datasets(df)
    if dir_ds is None:
        print("[RETRAIN] No directional rows found (BUY/SELL). Need some non-FLAT labels.")
        sys.exit(3)

    Xd, yd = dir_ds
    n_dir = int(Xd.shape[0])
    print(f"[RETRAIN] Directional rows: {n_dir} | features: {len(feat_cols)}")
    
    # Sanity: ensure SR features exist (optional warning)
    sr_expected = {
        "sr_1T_hi_dist", "sr_1T_lo_dist", "sr_3T_hi_dist", "sr_3T_lo_dist",
        "sr_5T_hi_dist", "sr_5T_lo_dist", "sr_breakout_up", "sr_breakout_dn"
    }
    missing = sorted([c for c in sr_expected if c not in feat_cols])
    if missing:
        print(f"[RETRAIN][WARN] SR features missing in dataset: {missing}")
        print("         The model will still train with available columns.")
    
    # Optional: require a minimum number of features (skip if you want)
    # if len(feat_cols) < 79:
    #     print(f"[RETRAIN][WARN] Only {len(feat_cols)} features found; 79 expected. Continuing.")

    # 3) Train XGB now (bypass online gate)
    bst = _train_xgb(Xd, yd)
    if bst is None:
        print("[RETRAIN] XGB training failed.")
        sys.exit(4)

    # 4) Embed feature schema inside booster
    schema = {"feature_names": list(feat_cols)}
    try:
        bst.set_attr(feature_schema=json.dumps(schema))
        print(f"[RETRAIN] Embedded feature_schema (n={len(feat_cols)})")
    except Exception as e:
        print(f"[RETRAIN][WARN] Failed to embed schema: {e}")

    # 5) Save booster to XGB_PATH
    try:
        os.makedirs(os.path.dirname(xgb_out_path), exist_ok=True)
        bst.save_model(xgb_out_path)
        print(f"[RETRAIN] Saved XGB to {xgb_out_path}")
    except Exception as e:
        print(f"[RETRAIN] Failed to save booster: {e}")
        sys.exit(5)

    # 6) Write a companion feature_schema.json (optional, helpful for tooling)
    try:
        schema_path = os.path.join(os.path.dirname(xgb_out_path), "feature_schema.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump({
                "feature_names": list(feat_cols),
                "num_features": len(feat_cols)
            }, f, indent=2)
        print(f"[RETRAIN] Wrote schema file: {schema_path}")
    except Exception as e:
        print(f"[RETRAIN][WARN] Failed to write schema file: {e}")

    # 7) Verify num_feature
    try:
        import xgboost as xgb
        b2 = xgb.Booster()
        b2.load_model(xgb_out_path)
        attrs = b2.attributes()
        print(f"[RETRAIN] Booster attributes: {list(attrs.keys()) if attrs else 'none'}")
        print(f"[RETRAIN] OK. Restart the app to load the new booster.")
    except Exception as e:
        print(f"[RETRAIN][WARN] Post-save check failed: {e}")


if __name__ == "__main__":
    main()
