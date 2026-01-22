#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for BacktestSummaryReport functionality.

Tests:
1. Year-by-year performance breakdown
2. Metadata extraction
3. CSV/Excel export with append mode
4. Integration with BacktestResult
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.metrics import (
    BacktestSummaryReport,
    BacktestMetadata,
    YearlyMetrics,
    PerformanceCalculator,
    Dimension,
    BacktestType,
    generate_summary_report,
)


def create_mock_equity_curve(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 1000000,
    annual_return: float = 0.15,
    volatility: float = 0.20
) -> pd.Series:
    """
    Create a mock equity curve for testing.

    Args:
        start_date: Start date string
        end_date: End date string
        initial_capital: Initial capital
        annual_return: Target annual return
        volatility: Target annual volatility

    Returns:
        pd.Series with DatetimeIndex
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    # Generate random returns with target characteristics
    daily_return = annual_return / 252
    daily_vol = volatility / np.sqrt(252)

    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(daily_return, daily_vol, n_days)

    # Convert to equity curve
    equity = initial_capital * np.cumprod(1 + returns)

    return pd.Series(equity, index=dates, name='equity')


def test_performance_calculator():
    """Test PerformanceCalculator year-by-year breakdown."""
    print("\n" + "=" * 60)
    print("Test 1: PerformanceCalculator Year-by-Year Breakdown")
    print("=" * 60)

    # Create mock equity curve spanning multiple years
    equity_curve = create_mock_equity_curve(
        start_date="2020-01-01",
        end_date="2023-12-31"
    )

    calculator = PerformanceCalculator()
    yearly_metrics = calculator.calculate_yearly_breakdown(
        equity_curve,
        include_total=True
    )

    print(f"\nEquity curve: {len(equity_curve)} days")
    print(f"Date range: {equity_curve.index[0]} to {equity_curve.index[-1]}")
    print(f"\nYearly Breakdown ({len(yearly_metrics)} periods):")
    print("-" * 80)
    print(f"{'Year':<8} {'Ann.Return':>12} {'Volatility':>12} {'Sharpe':>10} {'MaxDD':>10} {'WinRate':>10}")
    print("-" * 80)

    for m in yearly_metrics:
        print(f"{m.year:<8} {m.annualized_return:>11.2%} {m.volatility:>11.2%} "
              f"{m.sharpe_ratio:>10.3f} {m.max_drawdown:>9.2%} {m.win_rate:>9.2%}")

    assert len(yearly_metrics) == 5, "Should have 4 years + Total"
    assert yearly_metrics[-1].year == "Total", "Last entry should be Total"

    print("\n[PASS] PerformanceCalculator test passed!")
    return True


def test_summary_report_creation():
    """Test BacktestSummaryReport creation and export."""
    print("\n" + "=" * 60)
    print("Test 2: BacktestSummaryReport Creation")
    print("=" * 60)

    # Create report
    report = BacktestSummaryReport(output_dir="Reports/test")

    # Create mock equity curve
    equity_curve = create_mock_equity_curve()

    # Add backtest with metadata
    report.add_backtest(
        equity_curve=equity_curve,
        mode="single_factor_single_asset_ts",
        assets=["RB"],
        factors=["HurstExponent"],
        config_path="config/test_config.yaml"
    )

    # Get DataFrame
    df = report.to_dataframe()

    print(f"\nGenerated {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print("\nSample row:")
    if not df.empty:
        print(df.iloc[0].to_dict())

    assert len(df) > 0, "Should have generated rows"
    assert 'Year' in df.columns, "Should have Year column"
    assert 'Sharpe_Ratio' in df.columns, "Should have Sharpe_Ratio column"

    print("\n[PASS] Summary report creation test passed!")
    return True


def test_summary_report_export():
    """Test CSV and Excel export with append mode."""
    print("\n" + "=" * 60)
    print("Test 3: Summary Report Export (CSV & Excel)")
    print("=" * 60)

    import tempfile
    import os

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        report = BacktestSummaryReport(output_dir=tmpdir)

        # Add first backtest
        equity1 = create_mock_equity_curve(
            start_date="2020-01-01",
            end_date="2021-12-31",
            annual_return=0.10
        )

        report.add_backtest(
            equity_curve=equity1,
            mode="single_factor_single_asset_ts",
            assets=["RB"],
            factors=["HurstExponent"]
        )

        # Save to CSV
        csv_path = report.save("test_summary.csv", append=False, format='csv')
        print(f"\nSaved CSV: {csv_path}")

        # Verify CSV exists
        assert os.path.exists(csv_path), "CSV file should exist"

        # Read and verify
        df1 = pd.read_csv(csv_path)
        initial_rows = len(df1)
        print(f"Initial rows: {initial_rows}")

        # Clear and add second backtest
        report.clear()

        equity2 = create_mock_equity_curve(
            start_date="2020-01-01",
            end_date="2021-12-31",
            annual_return=0.20
        )

        report.add_backtest(
            equity_curve=equity2,
            mode="multi_factor_multi_asset_xs",
            assets=["RB", "HC", "I"],
            factors=["HurstExponent", "AMIHUDFactor"]
        )

        # Append to CSV
        csv_path = report.save("test_summary.csv", append=True, format='csv')

        # Verify append
        df2 = pd.read_csv(csv_path)
        print(f"After append: {len(df2)} rows")

        assert len(df2) > initial_rows, "Append should increase row count"

        # Test Excel export
        report.clear()
        report.add_backtest(
            equity_curve=equity1,
            mode="single_factor_single_pair",
            assets=["RB-HC"],
            factors=["CopulaPairFactor"]
        )

        try:
            xlsx_path = report.save("test_summary.xlsx", append=False, format='excel')
            print(f"Saved Excel: {xlsx_path}")
            assert os.path.exists(xlsx_path), "Excel file should exist"
        except ImportError:
            print("Note: openpyxl not installed, Excel export skipped")

    print("\n[PASS] Export test passed!")
    return True


def test_backtest_type_parsing():
    """Test BacktestType parsing from mode strings."""
    print("\n" + "=" * 60)
    print("Test 4: BacktestType Parsing")
    print("=" * 60)

    test_cases = [
        ("single_factor_single_asset_ts", BacktestType.SINGLE_FACTOR_TS),
        ("multi_factor_single_asset_ts", BacktestType.MULTI_FACTOR_TS),
        ("single_factor_multi_asset_xs", BacktestType.SINGLE_FACTOR_XS),
        ("multi_factor_multi_asset_xs", BacktestType.MULTI_FACTOR_XS),
        ("single_factor_single_pair", BacktestType.SINGLE_FACTOR_PAIR),
        ("multi_factor_single_pair", BacktestType.MULTI_FACTOR_PAIR),
    ]

    for mode_str, expected in test_cases:
        result = BacktestType.from_mode_string(mode_str)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {mode_str} -> {result.value} [{status}]")
        assert result == expected, f"Expected {expected}, got {result}"

    print("\n[PASS] BacktestType parsing test passed!")
    return True


def test_dimension_classification():
    """Test Dimension enum values."""
    print("\n" + "=" * 60)
    print("Test 5: Dimension Classification")
    print("=" * 60)

    print(f"  TIME_SERIES: {Dimension.TIME_SERIES.value}")
    print(f"  CROSS_SECTIONAL: {Dimension.CROSS_SECTIONAL.value}")
    print(f"  PAIR: {Dimension.PAIR.value}")

    assert Dimension.TIME_SERIES.value == "时序"
    assert Dimension.CROSS_SECTIONAL.value == "截面"
    assert Dimension.PAIR.value == "配对"

    print("\n[PASS] Dimension classification test passed!")
    return True


def test_convenience_function():
    """Test generate_summary_report convenience function."""
    print("\n" + "=" * 60)
    print("Test 6: Convenience Function")
    print("=" * 60)

    import tempfile
    import os

    equity_curve = create_mock_equity_curve()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "quick_summary.csv")

        result_path = generate_summary_report(
            equity_curve=equity_curve,
            output_path=output_path,
            mode="single_factor_single_asset_ts",
            assets=["RB"],
            factors=["HurstExponent"],
            append=False
        )

        print(f"\nGenerated report: {result_path}")
        assert os.path.exists(result_path), "Report should exist"

        df = pd.read_csv(result_path)
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

    print("\n[PASS] Convenience function test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CTASectorTrendV2 - Summary Report Test Suite")
    print("=" * 60)

    tests = [
        test_performance_calculator,
        test_summary_report_creation,
        test_summary_report_export,
        test_backtest_type_parsing,
        test_dimension_classification,
        test_convenience_function,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
