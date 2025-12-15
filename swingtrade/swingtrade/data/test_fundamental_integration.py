#!/usr/bin/env python3
"""
Test cases for fundamental integration (Option A POC).
Run: python test_fundamental_integration.py
"""

import logging
import sys
from pathlib import Path

# Add parent dir to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from data.fundamental_fetcher import SimpleFundamentalFetcher

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def test_fetcher_basic():
    """Test that SimpleFundamentalFetcher works"""
    fetcher = SimpleFundamentalFetcher()

    # Test fetch
    maruti = fetcher.fetch("MARUTI")
    assert maruti is not None, "MARUTI fetch failed"
    assert maruti["pat_yoy_pct"] == 8, f"Expected MARUTI PAT +8%, got {maruti['pat_yoy_pct']}"
    assert maruti["debt_to_equity"] == 0.8, f"Expected MARUTI D/E 0.8x, got {maruti['debt_to_equity']}"

    print("✅ test_fetcher_basic passed")


def test_axisbank_profit_crash():
    """AXISBANK profit -26% should be detected"""
    fetcher = SimpleFundamentalFetcher()

    axisbank = fetcher.fetch("AXISBANK")
    assert axisbank is not None, "AXISBANK fetch failed"
    assert axisbank["pat_yoy_pct"] == -26, f"Expected AXISBANK PAT -26%, got {axisbank['pat_yoy_pct']}"

    # This is a red flag - profit crash
    assert axisbank["pat_yoy_pct"] < -15, "AXISBANK should trigger profit crash alert"

    print("✅ test_axisbank_profit_crash passed")


def test_rrkabel_strong_growth():
    """RRKABEL profit +134.7% should be boosted"""
    fetcher = SimpleFundamentalFetcher()

    rrkabel = fetcher.fetch("RRKABEL")
    assert rrkabel is not None, "RRKABEL fetch failed"
    assert rrkabel["pat_yoy_pct"] == 134.7, f"Expected RRKABEL PAT +134.7%, got {rrkabel['pat_yoy_pct']}"

    # This should get a boost
    assert rrkabel["pat_yoy_pct"] > 20, "RRKABEL should trigger strong growth boost"

    print("✅ test_rrkabel_strong_growth passed")


def test_cholafin_leverage():
    """CHOLAFIN D/E 7.4x should be flagged"""
    fetcher = SimpleFundamentalFetcher()

    cholafin = fetcher.fetch("CHOLAFIN")
    assert cholafin is not None, "CHOLAFIN fetch failed"
    assert cholafin["debt_to_equity"] == 7.4, f"Expected CHOLAFIN D/E 7.4x, got {cholafin['debt_to_equity']}"

    # This is a warning - high leverage
    assert cholafin["debt_to_equity"] > 5, "CHOLAFIN should trigger high leverage alert"

    print("✅ test_cholafin_leverage passed")


def test_cache_freshness():
    """Test that cache freshness works"""
    fetcher = SimpleFundamentalFetcher()

    # First fetch (should cache)
    maruti1 = fetcher.fetch("MARUTI", max_age_hours=24)
    assert maruti1 is not None

    # Second fetch (should hit cache)
    maruti2 = fetcher.fetch("MARUTI", max_age_hours=24)
    assert maruti2 is not None
    assert maruti2 == maruti1

    print("✅ test_cache_freshness passed")


def test_missing_symbol():
    """Test that missing symbols return None gracefully"""
    fetcher = SimpleFundamentalFetcher()

    # Non-existent symbol
    result = fetcher.fetch("NOSUCHSTOCK")
    assert result is None, "Should return None for missing symbol"

    print("✅ test_missing_symbol passed")


if __name__ == "__main__":
    LOG.info("Running fundamental integration tests...")
    print("\n" + "=" * 80)

    try:
        test_fetcher_basic()
        test_axisbank_profit_crash()
        test_rrkabel_strong_growth()
        test_cholafin_leverage()
        test_cache_freshness()
        test_missing_symbol()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
