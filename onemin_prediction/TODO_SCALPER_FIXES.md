# Trend Capture TODO

- Relax setup gating in trend lanes: consider `REQUIRE_SETUP=0` for `lane=TREND` or reduce `PEN_NO_SETUP` when trend and flow agree.
- Revisit HTF veto thresholds/conditional override: tune `HTF_VETO_SOFT_FLOW_MIN`, `FLOW_STRONG_MIN`, `FLOW_VWAP_EXT`, `HTF_STRONG_VETO_MIN` for trend sessions.
- Review policy gating in trend regimes: inspect `POLICY_MIN_SUCCESS`, edge thresholds, and `policy_veto` behavior when the teacher is eligible.
