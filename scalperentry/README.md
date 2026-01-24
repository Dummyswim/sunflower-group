# TradeBrain + Futures Sidecar (Production Notes)

This bundle contains two cooperating programs:

- `tradebrain.py`: the main index-safe scalping brain that builds 1-minute candles from ticks, computes indicators on candle close (Lane A), and runs fast tick-time intent engines (Lane B).
- `futures_sidecar.py`: a separate process that subscribes to NIFTY futures via Dhan WebSocket v2 FULL packets, computes real participation signals (Volume/VWAP/CVD, optional OI + Depth), and writes them to a small CSV that `tradebrain.py` can read.

The design goal is to predict the next 1-minute candle with higher confidence by combining:
1) Index price structure (clean candle logic)
2) Futures participation/flow (real volume, and potentially OI + depth)

---

## How it runs (practically)

Terminal 1 (sidecar): connects to Dhan WS, subscribes to the futures instrument (one month), and continuously writes:

- `trained_models/production/fut_ticks_vwap_cvd.csv`
- `trained_models/production/fut_candles_vwap_cvd.csv` (this is what `tradebrain.py` reads)

Run:
```bash
python futures_sidecar.py
```

Terminal 2 (tradebrain): your main brain. On every 1-minute candle close it does a tiny tail-read of the last line of `fut_candles_vwap_cvd.csv` and attaches that snapshot into the emitted JSONL (and computes a simple `fut_cvd_delta`).

Run:
```bash
python tradebrain.py
```

If the sidecar is not running (or the file does not exist yet), `tradebrain.py` will not crash. It records `fut_flow: null` and continues.

---

## Key env vars

Sidecar:
```bash
export DHAN_CLIENT_ID="..."
export DHAN_ACCESS_TOKEN="..."
export FUT_EXCHANGE_SEGMENT="NSE_FNO"
export FUT_SECURITY_ID="..."           # one-month futures security_id
export FUT_OUTPUT_DIR="trained_models/production"
```

TradeBrain:
```bash
export TB_USE_FUT_FLOW="1"
export TB_FUT_SIDECAR_PATH="trained_models/production/fut_candles_vwap_cvd.csv"
export TB_FUT_FLOW_STALE_SEC="180"

# Optional JSONL paths
export TB_JSONL="tradebrain_signal.jsonl"
export TB_ARM_JSONL="tradebrain_arm.jsonl"
```

---

## 1. Architecture

### 1.1 Two-lane engine inside `tradebrain.py`

#### Lane A: 1-minute candle engine (slow truth)
Runs only when a minute closes.

- Accumulates ticks into a 1-minute candle.
- On candle close:
  - Appends candle to history.
  - Computes indicators/metrics.
  - Evaluates candle-based strategy.
  - Writes a JSONL signal row.

Why it exists: candle-close metrics are stable and reduce noise.

#### Lane B: Tick-time engines (fast intent)
Runs on every tick but throttled internally (~50ms).

When flat, it asks multiple engines for intent (arm/enter). Example:
- EMA915 engine: short horizon trend/momentum using EMA(9) and EMA(15) with angle/slope rules.
- Micro engine: very fast microstructure signals (velocity/accel, quick levels, etc).

A small resolver chooses which intent wins.

Why it exists: Lane A can be late; Lane B can be early.

---

### 1.2 Futures sidecar (`futures_sidecar.py`)

Runs as an independent process:

- Connects to Dhan WebSocket v2.
- Subscribes to the configured NIFTY futures instrument (NSE_FNO).
- Parses Quote (code=4) and Full (code=8) packets.
- Computes:
  - Session VWAP
  - dVol (incremental volume)
  - CVD (tick-rule signed volume approximation)
  - Optional extra from FULL packets:
    - total buy/sell qty
    - OI / OI high / OI low
    - 5-level market depth -> depth imbalance, spread, microprice

It writes:
- Per-tick CSV (for debugging/analysis)
- Per-minute CSV (consumed by TradeBrain)

---

## 2. Data flow between the two scripts

### 2.1 Outputs written by the sidecar

#### A) Per-tick CSV (`FUT_TICKS_PATH`)
Row format (no header):
1. ts (ISO8601, IST)
2. ltp
3. dvol
4. cum_vol
5. session_vwap
6. cvd
7. total_sell_qty (may be NaN if not available)
8. total_buy_qty
9. oi
10. depth_imb
11. spread
12. microprice

#### B) Per-minute CSV (`FUT_SIDECAR_PATH`)
Row format (no header):
1. ts (bucket start, ISO8601, IST)
2. open
3. high
4. low
5. close
6. vol (sum of dvol in candle)
7. ticks (tick count)
8. vwap (session vwap snapshot)
9. cvd (session cvd snapshot)
10. sell_qty (last snapshot in candle)
11. buy_qty
12. oi
13. oi_high_day
14. oi_low_day
15. doi (delta OI vs prior candle, if available)
16. depth_imb
17. spread
18. microprice

These extended columns are safe: if Dhan does not send them for a contract/packet type, they remain NaN.

---

### 2.2 How `tradebrain.py` consumes it

At each 1-minute candle close, `tradebrain.py`:

1) tails the last line of `TB_FUT_SIDECAR_PATH`
2) parses it with `_parse_fut_candle_row()`
3) rejects stale flow data using `TB_FUT_FLOW_STALE_SEC`
4) injects it into candle metrics under:

- `metrics["fut_flow"] = {...}`
- `metrics["fut_cvd_delta"] = cvd_now - last_cvd`

This makes futures participation available in every JSONL record for:
- offline analysis / pivoting
- future gating rules (if you decide to use it in live entries)

Note: In this build, futures flow is recorded into the JSON payload and available to strategy code, but not aggressively used to veto/force entries by default. This avoids creating new stacked veto traps until you validate thresholds.

---

## 3. What each module does

### 3.1 `tradebrain.py` (main brain)

#### 3.1.1 Core state
- `candles`: rolling deque of candle dicts (`max_candles`)
- `current_candle`: in-progress minute candle
- `pos`: active position state (if any)
- `pending_shock`: shock watch state (if enabled)

Thread safety:
- uses a single `_lock` for shared state
- file writes use dedicated locks for JSONL and ARM streams

#### 3.1.2 Candle metrics computed at close
Key metrics you will see in JSONL:

- HMA: smoothed anchor midline
- ATR: volatility scale
- Bands: `HMA +/- stretch_limit * ATR`
- Anchor distance: distance from midline in ATR units
- CLV: close-location value (range pressure)
- Body% / Wick%: doji filters
- Path efficiency: `abs(close-open)/total_travel`
- Smoothness: `abs(close-open)/sqrt(rv2)`
- pav_mult: normalized activity proxy (travel vs baseline)
- squeeze: range percentile contraction (range-based squeeze)

#### 3.1.3 CLV pressure proxy gating
CLV is geometric, so TradeBrain weights it by an activity proxy:

- activity_w is computed from `pav_mult` using `TB_ACTIVITY_W_LOW` -> `TB_ACTIVITY_W_HIGH` mapping.
- Confirmation checks that use CLV also require `activity_w >= TB_MIN_ACTIVITY_W_CONFIRM`.

This prevents pretty CLV candles from acting as pressure signals when tape activity is low.

#### 3.1.4 EMA915 engine (EMA 9/15 + angle rule)
The EMA915 engine:
- builds short bars (default 1s)
- updates EMA(9) and EMA(15)
- computes slopes and converts to angles in degrees
- requires angles >= `ema915_min_angle_deg` (default 30) and <= `ema915_max_angle_deg` (default 80)

Important: If you change `ema915_bar_ms`, you are changing the time scale of slope. Angle thresholds may need retuning.

#### 3.1.5 Micro engine (velocity/acceleration intent)
Micro engine computes velocity/acceleration using a real dt between evaluations:

- avoids fixed dt acceleration spikes
- uses guard rails (min dt, z-scores, etc)
- throttled by `_MS_EVAL_EVERY` to reduce CPU churn

#### 3.1.6 Signals, ARM stream, and JSONL layout
TradeBrain writes JSONL rows with:
- `suggestion` (HOLD / ARM_* / ENTRY_* / EXIT_*)
- `engine` (candle / ema915 / micro)
- candle metrics and context
- `stream` field:
  - `signal` for normal signals
  - `arm` for arm rows

ARM rows go to `tradebrain_arm.jsonl` by default, while normal signals go to `tradebrain_signal.jsonl`.

---

### 3.2 `futures_sidecar.py` (VWAP/CVD/OI/Depth sidecar)

#### 3.2.1 Packet handling
- Quote packet (code=4): LTP, LTQ, LTT, ATP, cumulative volume, etc.
- Full packet (code=8): includes above plus:
  - total sell qty, total buy qty
  - OI, OI high/low day (for NSE_FNO)
  - 5-level market depth

Parsing is defensive:
- checks packet length before reading optional fields
- depth parsing is wrapped and never stops the program if malformed

#### 3.2.2 Flow features computed
- dVol = max(0, cum_vol - last_cum_vol)
- VWAP = sum(price * dVol) / sum(dVol)
- CVD:
  - tick-rule sign from price change
  - `cum_cvd += sign * dVol`

Optional (FULL):
- depth_imb: (sum bid_qty - sum ask_qty) / (sum bid_qty + sum ask_qty)
- spread: best ask - best bid
- microprice: weighted by top-level quantities
- doi: OI change per minute candle

#### 3.2.3 Reliability and safety guards
- epoch timestamp conversion uses LTT with fallback to recv time
- sanity range checks for price
- reconnection loop with exponential backoff
- no infinite tight loops (attempt counter, backoff)
- file write errors are non-fatal (sidecar keeps running)

---

## 4. Running in production

### 4.1 Environment variables

Sidecar:
- `DHAN_ACCESS_TOKEN` (base64 encoded token as required by Dhan v2 ws)
- `DHAN_CLIENT_ID` (base64 encoded client id)
- `FUT_SECURITY_ID` (front-month NIFTY futures SecurityId)
- `FUT_EXCHANGE_SEGMENT` (default `NSE_FNO`)
- `FUT_TICKS_PATH` (default `trained_models/production/fut_ticks_vwap_cvd.csv`)
- `FUT_SIDECAR_PATH` (default `trained_models/production/fut_candles_vwap_cvd.csv`)

TradeBrain:
- `TB_USE_FUT_FLOW` (enable sidecar tail-read)
- `TB_FUT_SIDECAR_PATH` (sidecar candle CSV)
- `TB_FUT_FLOW_STALE_SEC` (stale cutoff in seconds)
- `TB_JSONL` (signal stream output)
- `TB_ARM_JSONL` (arm stream output)

### 4.2 Run

Sidecar:
```bash
python futures_sidecar.py
```

TradeBrain:
```bash
python tradebrain.py
```
