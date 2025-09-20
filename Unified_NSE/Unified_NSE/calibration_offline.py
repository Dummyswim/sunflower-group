import json, pathlib, math
from collections import defaultdict

def summarize_hitrate(jsonl_path="logs/hitrate.jsonl"):
    p = pathlib.Path(jsonl_path)
    if not p.exists():
        print(f"Missing: {jsonl_path}")
        return
    bins = defaultdict(lambda: {"n":0, "k":0})
    with p.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            dirn = str(row.get("direction","NEUTRAL")).upper()
            corr = row.get("correct", None)
            if dirn == "NEUTRAL" or corr is None:
                continue
            mtf = float(row.get("mtf_score",0.0))
            sr  = str(row.get("sr_room","UNKNOWN")).upper()
            band = "lt0.5" if mtf < 0.5 else "0.5-0.65" if mtf < 0.65 else "ge0.65"
            key = (band, sr)
            bins[key]["n"] += 1
            bins[key]["k"] += 1 if corr else 0
    print("Bucketed directional hit‑rates:")
    for k,v in sorted(bins.items()):
        n, k_ok = v["n"], v["k"]
        hr = (k_ok/n*100.0) if n else 0.0
        print(f"{k}: {hr:.1f}% ({k_ok}/{n})")

if __name__ == "__main__":
    summarize_hitrate()
