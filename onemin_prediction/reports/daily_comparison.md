# Daily Session Comparison

## Summary Snapshot
- Jan2: 30/56 (53.6%)
- Jan5: 34/64 (53.1%)
- Jan8: 35/70 (50.0%)
- Jan9: 36/69 (52.2%)
- Jan12: 26/70 (37.1%)
- Jan13: 24/64 (37.5%)

## Intent Mix
- Jan2: BUY 31, SELL 6, FLAT 21
- Jan5: BUY 11, SELL 20, FLAT 36
- Jan8: BUY 2, SELL 53, FLAT 17
- Jan9: BUY 6, SELL 46, FLAT 21
- Jan12: BUY 35, SELL 22, FLAT 13
- Jan13: BUY 21, SELL 25, FLAT 18

## Tradeable Rate
- Jan2: 1/58 (1.7%)
- Jan5: 1/67 (1.5%)
- Jan8: 2/72 (2.8%)
- Jan9: 1/73 (1.4%)
- Jan12: 37/70 (52.9%)
- Jan13: 27/64 (42.2%)

## Regime Mix
- Jan2: SIDEWAYS 48, CHOP 10
- Jan5: SIDEWAYS 27, TREND_UP 17, CHOP 3, TREND_DN 20
- Jan8: SIDEWAYS 16, TREND_DN 53, REVERSAL_RISK 3
- Jan9: SIDEWAYS 13, CHOP 9, TREND_DN 51
- Jan12: SIDEWAYS 21, TREND_DN 19, TREND_UP 30
- Jan13: SIDEWAYS 18, TREND_DN 23, CHOP 20, TREND_UP 3

## By-Label Accuracy
- Jan2:
  - BUY 13/16 (81%)
  - SELL 2/7 (29%)
  - FLAT 15/33 (45%)
- Jan5:
  - BUY 4/16 (25%)
  - SELL 11/20 (55%)
  - FLAT 19/28 (68%)
- Jan8:
  - BUY 1/13 (8%)
  - SELL 27/31 (87%)
  - FLAT 7/26 (27%)
- Jan9:
  - BUY 4/14 (29%)
  - SELL 23/30 (77%)
  - FLAT 9/25 (36%)
- Jan12:
  - BUY 12/24 (50%)
  - SELL 12/29 (41%)
  - FLAT 2/17 (12%)
- Jan13:
  - BUY 9/18 (50%)
  - SELL 12/32 (37.5%)
  - FLAT 3/14 (21%)

## Jan8 -> Jan9 Delta
- Accuracy: +2.2 points (50.0% -> 52.2%).
- BUY recall improved (8% -> 29%) but still low.
- SELL recall dipped (87% -> 77%).
- Tradeable rate down slightly (2.8% -> 1.4%).

## Jan9 -> Jan12 Delta
- Accuracy: -15.1 points (52.2% -> 37.1%).
- BUY recall improved (29% -> 50%), but SELL recall fell sharply (77% -> 41%).
- FLAT recall dropped (36% -> 12%).
- Tradeable rate jumped (1.4% -> 52.9%).

## Jan12 -> Jan13 Delta
- Accuracy: +0.4 points (37.1% -> 37.5%).
- BUY recall flat (50% -> 50%); SELL recall slipped (41% -> 37.5%).
- FLAT recall improved (12% -> 21%).
- Tradeable rate eased (52.9% -> 42.2%).

## Notes
- Intent swung from BUY-heavy (Jan2) to SELL-heavy (Jan8/Jan9).
- Tradeable rate spiked in Jan12/Jan13; verify whether gates or tradeable labeling shifted.
- Jan8/Jan9 dominated by TREND_DN regimes, correlating with SELL bias.

## Calibration Health (To Track Going Forward)
- Record brier_before_eval -> brier_after_eval for BUY/SELL after each session.
- Flag any session where brier_after_eval worsens; skip calibration for that session.
