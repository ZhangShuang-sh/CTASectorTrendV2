#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Single Factor Backtest Runner

Run all single-factor backtests and generate comprehensive summary report.

Backtest Types:
1. Single Factor - Single Asset - Time Series (时序因子)
2. Single Factor - Cross-Sectional (截面因子)
3. Single Factor - Single Pair (配对因子)

Output:
- Reports/backtest_results/backtest_summary.csv - Comprehensive summary
- Reports/backtest_results/plots/ - Visualization charts
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Configuration
# =============================================================================

# Backtest date range
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# Assets for time-series backtests
# Set to "ALL" to use all available assets from data, or specify a list
TS_TEST_ASSETS = "ALL"  # Will be populated from data
# TS_TEST_ASSETS = ["RB", "HC", "I", "J", "JM", "CU", "AL", "ZN", "AG", "AU"]  # Or specify subset

# Representative pairs for pair trading backtests
PAIR_TEST_PAIRS = [
    ("RB", "HC"),   # 螺纹钢-热卷
    ("I", "J"),     # 铁矿石-焦炭
    ("CU", "AL"),   # 铜-铝
    ("AG", "AU"),   # 白银-黄金
]

# Sectors for cross-sectional backtests
XS_TEST_SECTORS = ["Wind煤焦钢矿", "Wind有色"]

# Time series factors to test
TS_FACTORS = [
    {"name": "HurstExponent", "class": "core.factors.time_series.trend.HurstExponent", "category": "trend"},
    {"name": "EMDTrend", "class": "core.factors.time_series.trend.EMDTrend", "category": "trend"},
    {"name": "AMIHUDFactor", "class": "core.factors.time_series.liquidity.AMIHUDFactor", "category": "liquidity"},
    {"name": "AmivestFactor", "class": "core.factors.time_series.liquidity.AmivestFactor", "category": "liquidity"},
    {"name": "DUVOLFactor", "class": "core.factors.time_series.volatility_advanced.DUVOLFactor", "category": "volatility"},
    {"name": "KalmanFilterDeviation", "class": "core.factors.time_series.volatility.KalmanFilterDeviation", "category": "volatility"},
]

# Cross-sectional factors to test
XS_FACTORS = [
    {"name": "MomentumRank", "class": "core.factors.cross_sectional.momentum.MomentumRank", "category": "momentum"},
    {"name": "VolatilityFactor", "class": "core.factors.cross_sectional.price_volume.VolatilityFactor", "category": "volatility"},
    {"name": "LiquidityFactor", "class": "core.factors.cross_sectional.price_volume.LiquidityFactor", "category": "liquidity"},
]

# Pair trading factors to test
PAIR_FACTORS = [
    {"name": "CopulaPairFactor", "class": "core.factors.pair_trading.copula.CopulaPairFactor", "category": "copula"},
    {"name": "KalmanFilterPairFactor", "class": "core.factors.pair_trading.kalman_filter.KalmanFilterPairFactor", "category": "kalman"},
    {"name": "StyleMomentumPairFactor", "class": "core.factors.pair_trading.style_momentum.StyleMomentumPairFactor", "category": "momentum"},
]


# =============================================================================
# Data Loader
# =============================================================================

class SimpleDataLoader:
    """Simple data loader for backtesting."""

    def __init__(self, data_path: str = None):
        self.data_path = data_path or str(project_root / "data/raw/fut_primary.parquet")
        self._data_cache = None

    def load_data(self) -> pd.DataFrame:
        """Load futures data."""
        if self._data_cache is not None:
            return self._data_cache

        print(f"Loading data from: {self.data_path}")

        try:
            df = pd.read_parquet(self.data_path)
            print(f"Loaded {len(df)} rows")

            # Ensure date column
            if 'TRADE_DT' in df.columns:
                df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
                df = df.set_index('TRADE_DT')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

            self._data_cache = df
            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def get_asset_data(self, asset: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get data for a specific asset."""
        df = self.load_data()
        if df is None:
            return None

        # Filter by asset
        if 'PRODUCT_CODE' in df.columns:
            asset_df = df[df['PRODUCT_CODE'] == asset].copy()
        elif 'product' in df.columns:
            asset_df = df[df['product'] == asset].copy()
        else:
            print(f"Cannot find product column for {asset}")
            return None

        if asset_df.empty:
            return None

        # Filter by date
        if start_date:
            asset_df = asset_df[asset_df.index >= pd.Timestamp(start_date)]
        if end_date:
            asset_df = asset_df[asset_df.index <= pd.Timestamp(end_date)]

        # Add lowercase column aliases while keeping original column names
        # This ensures compatibility with factors that expect either format
        col_map = {
            'S_DQ_CLOSE': 'close',
            'S_DQ_OPEN': 'open',
            'S_DQ_HIGH': 'high',
            'S_DQ_LOW': 'low',
            'S_DQ_VOLUME': 'volume',
            'S_DQ_AMOUNT': 'amount',
        }
        for orig, alias in col_map.items():
            if orig in asset_df.columns:
                asset_df[alias] = asset_df[orig]  # Add alias, keep original

        return asset_df.sort_index()

    def get_all_assets(self) -> List[str]:
        """Get list of all available assets in the data."""
        df = self.load_data()
        if df is None:
            return []

        if 'PRODUCT_CODE' in df.columns:
            return sorted(df['PRODUCT_CODE'].unique().tolist())
        elif 'product' in df.columns:
            return sorted(df['product'].unique().tolist())
        return []

    def get_multi_asset_data(
        self,
        assets: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple assets."""
        df = self.load_data()
        if df is None:
            return {}

        result = {}

        # Get all available assets if not specified
        if assets is None:
            if 'PRODUCT_CODE' in df.columns:
                assets = df['PRODUCT_CODE'].unique().tolist()
            elif 'product' in df.columns:
                assets = df['product'].unique().tolist()
            else:
                return {}

        for asset in assets:
            asset_df = self.get_asset_data(asset, start_date, end_date)
            if asset_df is not None and not asset_df.empty:
                result[asset] = asset_df

        return result


# =============================================================================
# Simple Backtest Engine
# =============================================================================

class SimpleBacktester:
    """Simple backtester for single factor testing."""

    def __init__(
        self,
        initial_capital: float = 1000000,
        transaction_cost: float = 0.001,
        position_size: float = 1.0
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size

    def run_signal_backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        signal_threshold: float = 0.0
    ) -> pd.Series:
        """
        Run backtest based on signals.

        Args:
            prices: Price series
            signals: Signal series (positive = long, negative = short)
            signal_threshold: Threshold for signal to trigger position

        Returns:
            Equity curve
        """
        # Align signals and prices
        common_idx = prices.index.intersection(signals.index)
        if len(common_idx) < 10:
            return pd.Series(dtype=float)

        prices = prices.loc[common_idx]
        signals = signals.loc[common_idx]

        # Calculate returns
        returns = prices.pct_change().fillna(0)

        # Generate positions based on signals
        positions = pd.Series(0.0, index=signals.index)
        positions[signals > signal_threshold] = 1.0
        positions[signals < -signal_threshold] = -1.0

        # Shift positions to avoid look-ahead bias
        positions = positions.shift(1).fillna(0)

        # Calculate strategy returns
        strategy_returns = positions * returns

        # Apply transaction costs on position changes
        position_changes = positions.diff().abs().fillna(0)
        costs = position_changes * self.transaction_cost
        strategy_returns = strategy_returns - costs

        # Calculate equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        return equity

    def run_factor_backtest(
        self,
        factor_class: type,
        data: pd.DataFrame,
        factor_params: Dict = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Run backtest for a single factor.

        Args:
            factor_class: Factor class
            data: Asset data
            factor_params: Factor parameters

        Returns:
            (equity_curve, signals)
        """
        try:
            # Instantiate factor
            params = factor_params or {}
            factor = factor_class(**params)

            # Calculate factor values
            signals = pd.Series(index=data.index, dtype=float)

            # Use factor's own window size for rolling calculation
            window = getattr(factor, 'window', params.get('window', 60))

            for i in range(window, len(data)):
                try:
                    window_data = data.iloc[i-window:i+1]
                    result = factor.calculate(window_data)

                    if isinstance(result, (int, float)):
                        signals.iloc[i] = result
                    elif hasattr(result, 'values'):
                        signals.iloc[i] = float(result.values) if hasattr(result.values, '__float__') else 0.0
                    else:
                        signals.iloc[i] = 0.0

                except Exception:
                    signals.iloc[i] = 0.0

            # Get price series
            if 'close' in data.columns:
                prices = data['close']
            elif 'S_DQ_CLOSE' in data.columns:
                prices = data['S_DQ_CLOSE']
            else:
                return pd.Series(dtype=float), signals

            # Run backtest
            equity = self.run_signal_backtest(prices, signals)

            return equity, signals

        except Exception as e:
            print(f"  Error in factor backtest: {e}")
            return pd.Series(dtype=float), pd.Series(dtype=float)


# =============================================================================
# Visualization
# =============================================================================

def create_equity_plot(
    equity_curve: pd.Series,
    title: str,
    output_path: str
) -> bool:
    """Create and save equity curve plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        ax1 = axes[0]
        ax1.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=1)
        ax1.set_title(f'{title} - Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print(f"  Warning: Could not create plot: {e}")
        return False


# =============================================================================
# Main Backtest Runner
# =============================================================================

class SingleFactorBacktestRunner:
    """Run all single factor backtests."""

    def __init__(self):
        self.data_loader = SimpleDataLoader()
        self.backtester = SimpleBacktester()
        self.backtest_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Output directories
        self.output_dir = project_root / "Reports" / "backtest_results"
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = []

    def import_factor_class(self, class_path: str) -> Optional[type]:
        """Dynamically import factor class."""
        try:
            parts = class_path.rsplit('.', 1)
            if len(parts) != 2:
                return None

            module_path, class_name = parts
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)

        except Exception as e:
            print(f"  Could not import {class_path}: {e}")
            return None

    def run_ts_factor_backtest(
        self,
        factor_info: Dict,
        asset: str,
        data: pd.DataFrame
    ) -> Dict:
        """Run single time-series factor backtest."""
        factor_name = factor_info['name']
        factor_class = self.import_factor_class(factor_info['class'])

        if factor_class is None:
            return None

        print(f"  Running {factor_name} on {asset}...")

        try:
            equity, signals = self.backtester.run_factor_backtest(
                factor_class=factor_class,
                data=data
            )

            if equity.empty or len(equity) < 100:
                print(f"    Insufficient data for {factor_name}-{asset}")
                return None

            # Create visualization
            plot_filename = f"ts_{factor_name}_{asset}.png"
            plot_path = self.plots_dir / plot_filename
            create_equity_plot(
                equity,
                f"{factor_name} - {asset}",
                str(plot_path)
            )

            return {
                'equity_curve': equity,
                'dimension': '时序',
                'factor_layer_1': 'Common',
                'factor_layer_2': factor_info['category'],
                'asset_pool': asset,
                'backtest_type': '单因子-时序',
                'factor_name': factor_name,
                'start_date': str(equity.index[0].date()),
                'end_date': str(equity.index[-1].date()),
            }

        except Exception as e:
            print(f"    Error: {e}")
            return None

    def run_xs_factor_backtest(
        self,
        factor_info: Dict,
        multi_asset_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Run single cross-sectional factor backtest."""
        factor_name = factor_info['name']
        factor_class = self.import_factor_class(factor_info['class'])

        if factor_class is None:
            return None

        print(f"  Running {factor_name} (cross-sectional)...")

        try:
            # Simple cross-sectional backtest: rank assets and go long/short
            all_signals = {}

            for asset, data in multi_asset_data.items():
                try:
                    factor = factor_class()
                    window = 60

                    signals = pd.Series(index=data.index, dtype=float)
                    for i in range(window, len(data)):
                        try:
                            window_data = data.iloc[i-window:i+1]
                            result = factor.calculate(window_data)
                            if isinstance(result, (int, float)):
                                signals.iloc[i] = result
                        except Exception:
                            pass

                    all_signals[asset] = signals

                except Exception:
                    continue

            if len(all_signals) < 2:
                print(f"    Insufficient assets for cross-sectional")
                return None

            # Build cross-sectional portfolio equity
            # Simple approach: equal weight top/bottom quintile
            signals_df = pd.DataFrame(all_signals)
            returns_dict = {}
            for asset, data in multi_asset_data.items():
                if 'close' in data.columns:
                    returns_dict[asset] = data['close'].pct_change()
            returns_df = pd.DataFrame(returns_dict)

            # Align
            common_idx = signals_df.index.intersection(returns_df.index)
            signals_df = signals_df.loc[common_idx]
            returns_df = returns_df.loc[common_idx]

            # Cross-sectional rank
            ranks = signals_df.rank(axis=1, pct=True)

            # Long top 20%, short bottom 20%
            positions = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
            positions[ranks >= 0.8] = 1.0 / (ranks >= 0.8).sum(axis=1).replace(0, 1).values[:, None]
            positions[ranks <= 0.2] = -1.0 / (ranks <= 0.2).sum(axis=1).replace(0, 1).values[:, None]

            # Shift positions
            positions = positions.shift(1).fillna(0)

            # Portfolio returns
            portfolio_returns = (positions * returns_df).sum(axis=1)

            # Equity curve
            equity = self.backtester.initial_capital * (1 + portfolio_returns).cumprod()
            equity = equity.dropna()

            if len(equity) < 100:
                return None

            # Create visualization
            plot_filename = f"xs_{factor_name}.png"
            plot_path = self.plots_dir / plot_filename
            create_equity_plot(
                equity,
                f"{factor_name} - Cross-Sectional",
                str(plot_path)
            )

            assets_str = ", ".join(list(multi_asset_data.keys())[:5])
            if len(multi_asset_data) > 5:
                assets_str += f" ... ({len(multi_asset_data)} assets)"

            return {
                'equity_curve': equity,
                'dimension': '截面',
                'factor_layer_1': 'Common',
                'factor_layer_2': factor_info['category'],
                'asset_pool': assets_str,
                'backtest_type': '单因子-截面',
                'factor_name': factor_name,
                'start_date': str(equity.index[0].date()),
                'end_date': str(equity.index[-1].date()),
            }

        except Exception as e:
            print(f"    Error: {e}")
            return None

    def run_pair_factor_backtest(
        self,
        factor_info: Dict,
        pair: Tuple[str, str],
        data1: pd.DataFrame,
        data2: pd.DataFrame
    ) -> Dict:
        """Run single pair trading factor backtest."""
        factor_name = factor_info['name']
        factor_class = self.import_factor_class(factor_info['class'])

        if factor_class is None:
            return None

        asset1, asset2 = pair
        print(f"  Running {factor_name} on {asset1}-{asset2}...")

        try:
            # Align data
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) < 200:
                print(f"    Insufficient overlapping data")
                return None

            d1 = data1.loc[common_idx]
            d2 = data2.loc[common_idx]

            # Calculate pair signals
            factor = factor_class()
            signals = pd.Series(index=common_idx, dtype=float)
            window = 60

            for i in range(window, len(common_idx)):
                try:
                    w1 = d1.iloc[i-window:i+1]
                    w2 = d2.iloc[i-window:i+1]

                    # Some factors have calculate(data1, data2), others have different interface
                    if hasattr(factor, 'calculate'):
                        result = factor.calculate(w1, w2)
                        if isinstance(result, (int, float)):
                            signals.iloc[i] = result
                        elif hasattr(result, 'signal'):
                            signals.iloc[i] = result.signal
                        else:
                            signals.iloc[i] = 0.0
                except Exception:
                    signals.iloc[i] = 0.0

            # Simple pair trading: long spread when signal positive
            if 'close' in d1.columns:
                prices1 = d1['close']
                prices2 = d2['close']
            else:
                prices1 = d1['S_DQ_CLOSE']
                prices2 = d2['S_DQ_CLOSE']

            # Spread returns (long asset1, short asset2 when signal > 0)
            returns1 = prices1.pct_change()
            returns2 = prices2.pct_change()

            positions = pd.Series(0.0, index=signals.index)
            positions[signals > 0] = 1.0
            positions[signals < 0] = -1.0
            positions = positions.shift(1).fillna(0)

            # Pair spread return
            spread_returns = positions * (returns1 - returns2)

            equity = self.backtester.initial_capital * (1 + spread_returns).cumprod()
            equity = equity.dropna()

            if len(equity) < 100:
                return None

            # Create visualization
            plot_filename = f"pair_{factor_name}_{asset1}_{asset2}.png"
            plot_path = self.plots_dir / plot_filename
            create_equity_plot(
                equity,
                f"{factor_name} - {asset1}/{asset2}",
                str(plot_path)
            )

            return {
                'equity_curve': equity,
                'dimension': '配对',
                'factor_layer_1': 'Common',
                'factor_layer_2': factor_info['category'],
                'asset_pool': f"{asset1}-{asset2}",
                'backtest_type': '单因子-配对',
                'factor_name': factor_name,
                'start_date': str(equity.index[0].date()),
                'end_date': str(equity.index[-1].date()),
            }

        except Exception as e:
            print(f"    Error: {e}")
            return None

    def run_all(self):
        """Run all single factor backtests."""
        print("\n" + "=" * 60)
        print("CTASectorTrendV2 - Single Factor Backtest Suite")
        print("=" * 60)
        print(f"Backtest Time: {self.backtest_time}")
        print(f"Date Range: {START_DATE} to {END_DATE}")
        print("=" * 60)

        # Resolve asset list
        global TS_TEST_ASSETS
        if TS_TEST_ASSETS == "ALL":
            ts_assets = self.data_loader.get_all_assets()
            print(f"Using ALL {len(ts_assets)} available assets")
        else:
            ts_assets = TS_TEST_ASSETS
            print(f"Using {len(ts_assets)} specified assets")

        # Import summary report
        from core.metrics import BacktestSummaryReport
        report = BacktestSummaryReport(output_dir=str(self.output_dir))

        # =================================================================
        # 1. Time Series Factor Backtests
        # =================================================================
        print("\n[1/3] Running Time Series Factor Backtests...")
        print(f"Factors: {len(TS_FACTORS)}, Assets: {len(ts_assets)}")
        print(f"Total backtests: {len(TS_FACTORS) * len(ts_assets)}")
        print("-" * 60)

        for factor_info in TS_FACTORS:
            print(f"\nFactor: {factor_info['name']}")

            for asset in ts_assets:
                data = self.data_loader.get_asset_data(asset, START_DATE, END_DATE)
                if data is None or len(data) < 200:
                    print(f"  Skipping {asset}: insufficient data")
                    continue

                result = self.run_ts_factor_backtest(factor_info, asset, data)
                if result is not None:
                    report.add_backtest(
                        equity_curve=result['equity_curve'],
                        mode="single_factor_single_asset_ts",
                        assets=[result['asset_pool']],
                        factors=[result['factor_name']],
                        backtest_time=self.backtest_time
                    )
                    self.results.append(result)

        # =================================================================
        # 2. Cross-Sectional Factor Backtests
        # =================================================================
        print("\n[2/3] Running Cross-Sectional Factor Backtests...")
        print("-" * 60)

        # Load multi-asset data
        multi_asset_data = self.data_loader.get_multi_asset_data(
            assets=None,  # All available
            start_date=START_DATE,
            end_date=END_DATE
        )

        if len(multi_asset_data) >= 5:
            for factor_info in XS_FACTORS:
                print(f"\nFactor: {factor_info['name']}")

                result = self.run_xs_factor_backtest(factor_info, multi_asset_data)
                if result is not None:
                    report.add_backtest(
                        equity_curve=result['equity_curve'],
                        mode="single_factor_multi_asset_xs",
                        assets=list(multi_asset_data.keys()),
                        factors=[result['factor_name']],
                        backtest_time=self.backtest_time
                    )
                    self.results.append(result)
        else:
            print("  Insufficient multi-asset data for cross-sectional backtests")

        # =================================================================
        # 3. Pair Trading Factor Backtests
        # =================================================================
        print("\n[3/3] Running Pair Trading Factor Backtests...")
        print("-" * 60)

        for factor_info in PAIR_FACTORS:
            print(f"\nFactor: {factor_info['name']}")

            for pair in PAIR_TEST_PAIRS:
                asset1, asset2 = pair
                data1 = self.data_loader.get_asset_data(asset1, START_DATE, END_DATE)
                data2 = self.data_loader.get_asset_data(asset2, START_DATE, END_DATE)

                if data1 is None or data2 is None:
                    print(f"  Skipping {asset1}-{asset2}: missing data")
                    continue

                result = self.run_pair_factor_backtest(factor_info, pair, data1, data2)
                if result is not None:
                    report.add_backtest(
                        equity_curve=result['equity_curve'],
                        mode="single_factor_single_pair",
                        assets=[result['asset_pool']],
                        factors=[result['factor_name']],
                        backtest_time=self.backtest_time
                    )
                    self.results.append(result)

        # =================================================================
        # Save Summary Report
        # =================================================================
        print("\n" + "=" * 60)
        print("Saving Results...")
        print("=" * 60)

        # Save as new file (not append)
        summary_path = report.save(
            "backtest_summary.csv",
            append=False,
            format='csv'
        )

        # Also save Excel version
        try:
            report.save(
                "backtest_summary.xlsx",
                append=False,
                format='excel'
            )
        except Exception as e:
            print(f"Note: Excel export skipped: {e}")

        # Print summary
        print(f"\nTotal backtests completed: {len(self.results)}")
        print(f"Summary report: {summary_path}")
        print(f"Plots directory: {self.plots_dir}")

        # Print summary statistics
        stats = report.get_summary_stats()
        if stats:
            print(f"\nSummary Statistics (Total period):")
            print(f"  Average Sharpe: {stats.get('avg_sharpe', 0):.3f}")
            print(f"  Average Return: {stats.get('avg_return', 0):.2%}")
            print(f"  Average Max DD: {stats.get('avg_max_dd', 0):.2%}")

        print("\n" + "=" * 60)
        print("Backtest Complete!")
        print("=" * 60)

        return report


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    runner = SingleFactorBacktestRunner()
    runner.run_all()
