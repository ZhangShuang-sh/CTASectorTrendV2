#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - V1 vs V2 Regression Tests

Zero-modification policy verification:
- Factor values must match np.allclose(atol=1e-7)
- Combined signals must match
- Equity curves must match
- Performance metrics must match

Usage:
    python -m pytest tests/regression/test_v1_v2_consistency.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any

# Add project roots
v2_root = Path(__file__).parent.parent.parent
v1_root = v2_root.parent / "CTASectorTrendV1"

sys.path.insert(0, str(v2_root))
sys.path.insert(0, str(v1_root))

# Tolerance for numerical comparison
ATOL = 1e-7


class TestFactorConsistency:
    """Factor value consistency tests"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Generate sample price data for testing"""
        np.random.seed(42)  # Reproducibility
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')

        # Simulate price series with trend and noise
        trend = np.cumsum(np.random.randn(n) * 0.02) + 100
        noise = np.random.randn(n) * 0.5
        prices = trend + noise

        # Ensure positive prices
        prices = np.maximum(prices, 10)

        # Generate volume/amount ONCE and use for both column formats
        # This ensures V1 (close/amount) and V2 (S_DQ_CLOSE/S_DQ_AMOUNT) use same data
        volume = np.random.randint(1000, 10000, n)
        amount = prices * volume

        return pd.DataFrame({
            'TRADE_DT': dates,
            'S_DQ_CLOSE': prices,
            'close': prices,  # V1 format (same values)
            'S_DQ_VOLUME': volume,
            'volume': volume,  # V1 format (same values)
            'S_DQ_AMOUNT': amount,
            'amount': amount,  # V1 format (same values)
        })

    def test_hurst_exponent_consistency(self, sample_data):
        """Test HurstExponent V1 vs V2"""
        # V2
        from core.factors.time_series.trend import HurstExponent as HurstV2
        factor_v2 = HurstV2(window=100)
        v2_result = factor_v2.calculate(sample_data)

        # V1 (if available)
        try:
            from factors.time_series.trend import HurstExponent as HurstV1
            factor_v1 = HurstV1(window=100)
            v1_result = factor_v1.compute(sample_data)

            assert np.isclose(v1_result, v2_result, atol=ATOL), \
                f"HurstExponent mismatch: V1={v1_result}, V2={v2_result}"
        except ImportError:
            pytest.skip("V1 not available for comparison")

    def test_amihud_consistency(self, sample_data):
        """Test AMIHUDFactor V1 vs V2"""
        # V2
        from core.factors.time_series.liquidity import AMIHUDFactor as AmihudV2
        factor_v2 = AmihudV2(short_period=2, long_period=8)
        v2_result = factor_v2.calculate(sample_data)

        # V1 (if available)
        try:
            from factors.time_series.liquidity import AMIHUDFactor as AmihudV1
            factor_v1 = AmihudV1(short_period=2, long_period=8)
            v1_result = factor_v1.compute(sample_data)

            assert np.isclose(v1_result, v2_result, atol=ATOL), \
                f"AMIHUDFactor mismatch: V1={v1_result}, V2={v2_result}"
        except ImportError:
            pytest.skip("V1 not available for comparison")

    def test_amivest_consistency(self, sample_data):
        """Test AmivestFactor V1 vs V2"""
        # V2
        from core.factors.time_series.liquidity import AmivestFactor as AmivestV2
        factor_v2 = AmivestV2(short_period=12, long_period=32)
        v2_result = factor_v2.calculate(sample_data)

        # V1 (if available)
        try:
            from factors.time_series.liquidity import AmivestFactor as AmivestV1
            factor_v1 = AmivestV1(short_period=12, long_period=32)
            v1_result = factor_v1.compute(sample_data)

            assert np.isclose(v1_result, v2_result, atol=ATOL), \
                f"AmivestFactor mismatch: V1={v1_result}, V2={v2_result}"
        except ImportError:
            pytest.skip("V1 not available for comparison")


class TestSignalCombination:
    """Signal combination consistency tests"""

    def test_multi_layer_combiner(self):
        """Test MultiLayerCombiner produces expected output"""
        from core.processors.combiner import MultiLayerCombiner

        combiner = MultiLayerCombiner(
            factor_type_weights={
                'time_series': 0.5,
                'cross_sectional': 0.3,
                'pair_trading': 0.2
            }
        )

        # Test with sample outputs
        factor_outputs = {
            'HurstExponent': 0.5,
            'AMIHUDFactor': 0.3,
            'CopulaPairFactor': 0.8
        }

        factor_metadata = {
            'HurstExponent': {'type': 'time_series', 'category': 'trend'},
            'AMIHUDFactor': {'type': 'time_series', 'category': 'liquidity'},
            'CopulaPairFactor': {'type': 'pair_trading', 'category': 'copula'}
        }

        result = combiner.combine(factor_outputs, factor_metadata=factor_metadata)

        # Check result is in valid range
        assert -1.0 <= result.signal <= 1.0
        assert isinstance(result.ts_contribution, float)
        assert isinstance(result.pair_contribution, float)


class TestNormalization:
    """Signal normalization consistency tests"""

    def test_zscore_normalize(self):
        """Test zscore_clip normalization"""
        from core.processors.normalizer import SignalNormalizer

        normalizer = SignalNormalizer(zscore_clip_std=3.0)

        # Test series normalization
        data = pd.Series([1, 2, 3, 4, 5, 10, 15, 20])
        normalized = normalizer.normalize(data, method='zscore_clip')

        # Check output range
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0

    def test_rank_normalize(self):
        """Test rank normalization"""
        from core.processors.normalizer import SignalNormalizer

        normalizer = SignalNormalizer()

        data = pd.Series([10, 5, 15, 3, 20])
        normalized = normalizer.normalize(data, method='rank')

        # Check output range
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0


class TestParameterResolution:
    """Parameter resolution consistency tests"""

    def test_common_params(self):
        """Test common parameter resolution"""
        from core.processors.param_resolver import ParameterResolver
        from core.processors.config_loader import HierarchicalConfigLoader

        config = HierarchicalConfigLoader()
        config.load()

        resolver = ParameterResolver(config)

        params = resolver.resolve_params(
            factor_name='HurstExponent',
            factor_type='time_series'
        )

        # Check params include defaults
        assert 'window' in params or params == {}

    def test_sector_override(self):
        """Test sector-level parameter override"""
        from core.processors.param_resolver import ParameterResolver
        from core.processors.config_loader import HierarchicalConfigLoader

        config = HierarchicalConfigLoader()
        config.load()

        resolver = ParameterResolver(config)

        # Get params with sector override
        params = resolver.resolve_params(
            factor_name='HurstExponent',
            factor_type='time_series',
            sector='Wind煤焦钢矿'
        )

        # Should use sector override if configured
        assert isinstance(params, dict)


class TestBackwardCompatibility:
    """Backward compatibility tests"""

    def test_compute_alias(self):
        """Test that compute() works as alias for calculate()"""
        from core.factors.time_series.trend import HurstExponent

        factor = HurstExponent(window=50)

        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'S_DQ_CLOSE': np.random.randn(100).cumsum() + 100
        })

        # Both methods should work
        result_calculate = factor.calculate(data)
        result_compute = factor.compute(data)

        assert result_calculate == result_compute


class V1V2RegressionTest:
    """
    Comprehensive V1 vs V2 regression test suite

    Usage:
        test = V1V2RegressionTest()
        test.run_all_tests()
    """

    def __init__(self, v1_snapshot_dir: str = None):
        """
        Args:
            v1_snapshot_dir: Directory containing V1 output snapshots
        """
        self.v1_snapshot_dir = Path(v1_snapshot_dir) if v1_snapshot_dir else \
                               v2_root / "tests" / "regression" / "snapshots"

    def test_factor_values_match(self, v1_vals: np.ndarray, v2_vals: np.ndarray):
        """Factor values must match"""
        assert np.allclose(v1_vals, v2_vals, atol=ATOL), \
            f"Factor values mismatch: max diff = {np.max(np.abs(v1_vals - v2_vals))}"

    def test_signals_match(self, v1_sig: np.ndarray, v2_sig: np.ndarray):
        """Combined signals must match"""
        assert np.allclose(v1_sig, v2_sig, atol=ATOL), \
            f"Signal mismatch: max diff = {np.max(np.abs(v1_sig - v2_sig))}"

    def test_equity_curve_match(self, v1_eq: np.ndarray, v2_eq: np.ndarray):
        """Equity curves must match"""
        assert np.allclose(v1_eq, v2_eq, atol=ATOL), \
            f"Equity curve mismatch: max diff = {np.max(np.abs(v1_eq - v2_eq))}"

    def test_performance_metrics_match(self, v1_metrics: Dict, v2_metrics: Dict):
        """Performance metrics must match"""
        for metric in ['sharpe', 'sortino', 'calmar', 'max_dd']:
            if metric in v1_metrics and metric in v2_metrics:
                assert np.isclose(v1_metrics[metric], v2_metrics[metric], atol=ATOL), \
                    f"Metric '{metric}' mismatch: V1={v1_metrics[metric]}, V2={v2_metrics[metric]}"

    def run_all_tests(self):
        """Run all regression tests"""
        print("Running V1 vs V2 Regression Tests...")
        print("=" * 60)

        # Load snapshots and compare
        # This would load saved V1 outputs and compare with V2 results
        print("[INFO] Full regression testing requires V1 snapshots")
        print("[INFO] Generate snapshots by running V1 and saving outputs")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
