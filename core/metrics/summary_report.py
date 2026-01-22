#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Backtest Summary Report Module

Generates comprehensive CSV/Excel reports aggregating performance metrics
across different dimensions with year-by-year breakdown for manual factor
blending analysis.

Report Structure:
- Dimension: Cross-sectional (截面), Time-series (时序), Pair (配对)
- Factor Layer 1: Common/Idiosyncratic
- Factor Layer 2: Logical Category
- Asset Pool: Individual assets or basket string
- Backtest Type: Single/Multi-Factor × Time-series/Cross-sectional/Pair
- Factor Name: Factor name(s) with config path
- Year: Specific year or "Total"
- Performance Metrics: Ann.Return, Volatility, Sharpe, Calmar, MaxDD, WinRate
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
import os


class Dimension(Enum):
    """Factor dimension classification."""
    TIME_SERIES = "时序"
    CROSS_SECTIONAL = "截面"
    PAIR = "配对"


class BacktestType(Enum):
    """Backtest type classification (6 types)."""
    SINGLE_FACTOR_TS = "单因子-时序"
    MULTI_FACTOR_TS = "多因子-时序"
    SINGLE_FACTOR_XS = "单因子-截面"
    MULTI_FACTOR_XS = "多因子-截面"
    SINGLE_FACTOR_PAIR = "单因子-配对"
    MULTI_FACTOR_PAIR = "多因子-配对"

    @classmethod
    def from_mode_string(cls, mode: str) -> 'BacktestType':
        """Parse backtest type from mode string."""
        mode_lower = mode.lower()

        is_single = 'single_factor' in mode_lower
        is_multi = 'multi_factor' in mode_lower
        is_ts = mode_lower.endswith('_ts')
        is_xs = mode_lower.endswith('_xs')
        is_pair = 'pair' in mode_lower

        if is_single and is_ts:
            return cls.SINGLE_FACTOR_TS
        elif is_multi and is_ts:
            return cls.MULTI_FACTOR_TS
        elif is_single and is_xs:
            return cls.SINGLE_FACTOR_XS
        elif is_multi and is_xs:
            return cls.MULTI_FACTOR_XS
        elif is_single and is_pair:
            return cls.SINGLE_FACTOR_PAIR
        elif is_multi and is_pair:
            return cls.MULTI_FACTOR_PAIR
        elif 'comprehensive' in mode_lower:
            return cls.MULTI_FACTOR_XS  # Default for comprehensive

        # Default based on partial match
        if is_pair:
            return cls.SINGLE_FACTOR_PAIR if is_single else cls.MULTI_FACTOR_PAIR
        elif is_xs:
            return cls.SINGLE_FACTOR_XS if is_single else cls.MULTI_FACTOR_XS
        else:
            return cls.SINGLE_FACTOR_TS if is_single else cls.MULTI_FACTOR_TS


@dataclass
class BacktestMetadata:
    """
    Metadata extracted from backtest configuration.

    Used to populate report columns beyond performance metrics.
    """
    # Dimension classification
    dimension: str = ""

    # L1: Factor scope
    factor_layer_1: str = ""  # Common / Idiosyncratic / Mixed

    # L2: Logical category
    factor_layer_2: str = ""  # trend, volatility, liquidity, momentum, etc.

    # Asset information
    asset_pool: str = ""  # Individual asset codes or basket description

    # Backtest type
    backtest_type: str = ""  # One of 6 types

    # Factor information
    factor_names: List[str] = field(default_factory=list)
    config_path: str = ""  # Path to config file used

    # Additional metadata
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0

    def get_factor_name_string(self) -> str:
        """
        Get formatted factor name string.

        For single factor: returns factor name
        For multi factor: returns comma-separated list + config path
        """
        if not self.factor_names:
            return "Unknown"

        if len(self.factor_names) == 1:
            return self.factor_names[0]
        else:
            names = ", ".join(self.factor_names)
            if self.config_path:
                return f"{names} (Config: {self.config_path})"
            return names


@dataclass
class YearlyMetrics:
    """Performance metrics for a specific period."""
    year: str  # "2020", "2021", ... or "Total"
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    sortino_ratio: float = 0.0
    total_return: float = 0.0
    trading_days: int = 0


class PerformanceCalculator:
    """
    Calculate performance metrics from equity curve.

    Supports both full-period and year-by-year calculations.
    """

    def __init__(self, annualization_factor: int = 252):
        self.annualization_factor = annualization_factor

    def calculate_yearly_breakdown(
        self,
        equity_curve: pd.Series,
        include_total: bool = True
    ) -> List[YearlyMetrics]:
        """
        Calculate performance metrics for each year and total period.

        Args:
            equity_curve: Daily equity curve with DatetimeIndex
            include_total: Whether to include total period metrics

        Returns:
            List of YearlyMetrics for each year + total
        """
        results = []

        if equity_curve is None or equity_curve.empty:
            return results

        # Ensure DatetimeIndex
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            try:
                equity_curve.index = pd.to_datetime(equity_curve.index)
            except Exception:
                return results

        # Sort by date
        equity_curve = equity_curve.sort_index()

        # Get unique years
        years = equity_curve.index.year.unique()

        # Calculate metrics for each year
        for year in sorted(years):
            year_mask = equity_curve.index.year == year
            year_equity = equity_curve[year_mask]

            if len(year_equity) < 2:
                continue

            metrics = self._calculate_metrics(year_equity, str(year))
            results.append(metrics)

        # Calculate total period metrics
        if include_total and len(equity_curve) >= 2:
            total_metrics = self._calculate_metrics(equity_curve, "Total")
            results.append(total_metrics)

        return results

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        period_label: str
    ) -> YearlyMetrics:
        """
        Calculate all performance metrics for a given equity curve.

        Args:
            equity_curve: Equity curve for the period
            period_label: Label for this period (year or "Total")

        Returns:
            YearlyMetrics object
        """
        if len(equity_curve) < 2:
            return YearlyMetrics(year=period_label)

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) < 1:
            return YearlyMetrics(year=period_label)

        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized return
        n_days = len(returns)
        n_years = n_days / self.annualization_factor

        if n_years > 0:
            ann_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            ann_return = total_return

        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(self.annualization_factor)

        # Sharpe ratio
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        # Sortino ratio (downside volatility)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(self.annualization_factor)
            sortino = ann_return / downside_vol if downside_vol > 0 else 0.0
        else:
            sortino = float('inf') if ann_return > 0 else 0.0

        # Max drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = abs(drawdown.min()) if not drawdown.empty else 0.0

        # Calmar ratio
        calmar = ann_return / max_dd if max_dd > 0 else 0.0

        # Win rate
        win_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = win_days / total_days if total_days > 0 else 0.0

        return YearlyMetrics(
            year=period_label,
            annualized_return=ann_return,
            volatility=ann_vol,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            sortino_ratio=sortino,
            total_return=total_return,
            trading_days=n_days
        )


class BacktestSummaryReport:
    """
    Generate comprehensive backtest summary reports.

    Features:
    - Year-by-year performance breakdown
    - Support for multiple dimensions (TS, XS, Pair)
    - Append mode for comparing multiple backtests
    - CSV and Excel export

    Usage:
        report = BacktestSummaryReport()

        # Add backtest results
        report.add_backtest(
            result=backtest_result,
            metadata=metadata,
            config=task_config
        )

        # Save to file
        report.save("Reports/backtest_summary.csv")
    """

    # Column definitions
    COLUMNS = [
        'Dimension',           # 时序/截面/配对
        'Factor_Layer_1',      # Common/Idiosyncratic
        'Factor_Layer_2',      # trend/volatility/liquidity/momentum/copula/etc
        'Asset_Pool',          # Asset codes or basket description
        'Backtest_Type',       # 6 types
        'Factor_Name',         # Factor name(s) + config path
        'Year',                # 2020, 2021, ... or Total
        'Annualized_Return',   # 年化收益率
        'Volatility',          # 年化波动率
        'Sharpe_Ratio',        # 夏普比率
        'Calmar_Ratio',        # 卡玛比率
        'Max_Drawdown',        # 最大回撤
        'Win_Rate',            # 胜率
        'Sortino_Ratio',       # 索提诺比率
        'Total_Return',        # 总收益率
        'Trading_Days',        # 交易天数
        'Start_Date',          # 回测开始日期
        'End_Date',            # 回测结束日期
        'Config_Path',         # 配置文件路径
        'Generated_At',        # 报告生成时间
    ]

    def __init__(self, output_dir: str = "Reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Default output directory
        """
        self.output_dir = Path(output_dir)
        self.calculator = PerformanceCalculator()
        self._rows: List[Dict] = []

    def add_backtest(
        self,
        equity_curve: pd.Series,
        metadata: BacktestMetadata = None,
        config: Dict = None,
        mode: str = None,
        assets: List[str] = None,
        factors: List[str] = None,
        config_path: str = None
    ) -> None:
        """
        Add a backtest result to the report.

        Args:
            equity_curve: Daily equity curve (DatetimeIndex)
            metadata: Pre-populated BacktestMetadata object
            config: Task configuration dictionary (alternative to metadata)
            mode: Backtest mode string (if not using metadata)
            assets: Asset list (if not using metadata)
            factors: Factor names list (if not using metadata)
            config_path: Configuration file path
        """
        # Build metadata if not provided
        if metadata is None:
            metadata = self._extract_metadata(
                config=config,
                mode=mode,
                assets=assets,
                factors=factors,
                config_path=config_path
            )

        # Calculate yearly breakdown
        yearly_metrics = self.calculator.calculate_yearly_breakdown(
            equity_curve,
            include_total=True
        )

        # Generate timestamp
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create rows for each year
        for metrics in yearly_metrics:
            row = {
                'Dimension': metadata.dimension,
                'Factor_Layer_1': metadata.factor_layer_1,
                'Factor_Layer_2': metadata.factor_layer_2,
                'Asset_Pool': metadata.asset_pool,
                'Backtest_Type': metadata.backtest_type,
                'Factor_Name': metadata.get_factor_name_string(),
                'Year': metrics.year,
                'Annualized_Return': metrics.annualized_return,
                'Volatility': metrics.volatility,
                'Sharpe_Ratio': metrics.sharpe_ratio,
                'Calmar_Ratio': metrics.calmar_ratio,
                'Max_Drawdown': metrics.max_drawdown,
                'Win_Rate': metrics.win_rate,
                'Sortino_Ratio': metrics.sortino_ratio,
                'Total_Return': metrics.total_return,
                'Trading_Days': metrics.trading_days,
                'Start_Date': metadata.start_date,
                'End_Date': metadata.end_date,
                'Config_Path': metadata.config_path,
                'Generated_At': generated_at,
            }
            self._rows.append(row)

    def add_from_backtest_result(
        self,
        result: Any,  # BacktestResult
        task_config: Any = None,  # TaskConfig
        config_path: str = None
    ) -> None:
        """
        Add backtest result from BacktestResult object.

        Args:
            result: BacktestResult object
            task_config: TaskConfig object (optional)
            config_path: Configuration file path
        """
        # Extract equity curve
        equity_curve = getattr(result, 'equity_curve', None)

        if equity_curve is None or (hasattr(equity_curve, 'empty') and equity_curve.empty):
            print("Warning: Empty equity curve, skipping")
            return

        # Extract metadata from task config
        metadata = self._extract_metadata_from_task_config(
            task_config,
            config_path
        ) if task_config else None

        # Add to report
        self.add_backtest(
            equity_curve=equity_curve,
            metadata=metadata,
            config_path=config_path
        )

    def _extract_metadata(
        self,
        config: Dict = None,
        mode: str = None,
        assets: List[str] = None,
        factors: List[str] = None,
        config_path: str = None
    ) -> BacktestMetadata:
        """
        Extract metadata from configuration.

        Args:
            config: Task configuration dictionary
            mode: Backtest mode string
            assets: Asset list
            factors: Factor names list
            config_path: Configuration file path

        Returns:
            BacktestMetadata object
        """
        metadata = BacktestMetadata()

        # Parse from config dict
        if config:
            mode = config.get('mode', mode)
            assets = config.get('assets', {}).get('symbols', assets)
            factors_config = config.get('factors', {})

            # Extract factor names from all categories
            factor_names = []
            for category in ['time_series', 'cross_sectional', 'pair_trading']:
                cat_factors = factors_config.get(category, [])
                for f in cat_factors:
                    if f.get('enabled', True):
                        factor_names.append(f.get('name', 'Unknown'))

            if factor_names:
                factors = factor_names

            # Date range
            date_range = config.get('date_range', {})
            metadata.start_date = date_range.get('start', '')
            metadata.end_date = date_range.get('end', '')

            # Execution config
            exec_config = config.get('execution', {})
            metadata.initial_capital = exec_config.get('initial_capital', 10000000)

        # Determine dimension
        if mode:
            mode_lower = mode.lower()
            if 'pair' in mode_lower:
                metadata.dimension = Dimension.PAIR.value
            elif 'xs' in mode_lower or 'cross' in mode_lower:
                metadata.dimension = Dimension.CROSS_SECTIONAL.value
            else:
                metadata.dimension = Dimension.TIME_SERIES.value

            # Determine backtest type
            metadata.backtest_type = BacktestType.from_mode_string(mode).value

        # Set factor layer 1 (Common/Idiosyncratic)
        # Default to Common if not specified
        metadata.factor_layer_1 = "Common"

        # Set factor layer 2 (category)
        if factors:
            # Infer category from factor names
            categories = self._infer_categories(factors)
            metadata.factor_layer_2 = ", ".join(categories) if categories else "mixed"

        # Asset pool
        if assets:
            if len(assets) <= 5:
                metadata.asset_pool = ", ".join(assets)
            else:
                metadata.asset_pool = f"{', '.join(assets[:3])} ... ({len(assets)} assets)"

        # Factor names
        if factors:
            metadata.factor_names = factors

        # Config path
        if config_path:
            metadata.config_path = config_path

        return metadata

    def _extract_metadata_from_task_config(
        self,
        task_config: Any,
        config_path: str = None
    ) -> BacktestMetadata:
        """
        Extract metadata from TaskConfig object.

        Args:
            task_config: TaskConfig object
            config_path: Configuration file path

        Returns:
            BacktestMetadata object
        """
        metadata = BacktestMetadata()

        # Mode
        mode = getattr(task_config, 'mode', None)
        if mode:
            mode_value = mode.value if hasattr(mode, 'value') else str(mode)

            if 'pair' in mode_value.lower():
                metadata.dimension = Dimension.PAIR.value
            elif 'xs' in mode_value.lower():
                metadata.dimension = Dimension.CROSS_SECTIONAL.value
            else:
                metadata.dimension = Dimension.TIME_SERIES.value

            metadata.backtest_type = BacktestType.from_mode_string(mode_value).value

        # Assets
        assets = getattr(task_config, 'assets', [])
        if assets:
            if len(assets) <= 5:
                metadata.asset_pool = ", ".join(assets)
            else:
                metadata.asset_pool = f"{', '.join(assets[:3])} ... ({len(assets)} assets)"

        # Pairs (for pair trading)
        pairs = getattr(task_config, 'pairs', [])
        if pairs and metadata.dimension == Dimension.PAIR.value:
            pair_strs = [f"{p[0]}-{p[1]}" for p in pairs[:3]]
            if len(pairs) > 3:
                metadata.asset_pool = f"{', '.join(pair_strs)} ... ({len(pairs)} pairs)"
            else:
                metadata.asset_pool = ", ".join(pair_strs)

        # Factors
        factors_config = getattr(task_config, 'factors', {})
        factor_names = []
        categories = set()

        for category in ['time_series', 'cross_sectional', 'pair_trading']:
            cat_factors = factors_config.get(category, [])
            for f in cat_factors:
                if f.get('enabled', True):
                    factor_names.append(f.get('name', 'Unknown'))
                    # Try to get category from factor config
                    cat = f.get('category', category)
                    categories.add(cat)

        metadata.factor_names = factor_names
        metadata.factor_layer_2 = ", ".join(categories) if categories else "mixed"

        # Factor layer 1
        metadata.factor_layer_1 = "Common"  # Default

        # Dates
        start_date = getattr(task_config, 'start_date', None)
        end_date = getattr(task_config, 'end_date', None)
        metadata.start_date = str(start_date) if start_date else ""
        metadata.end_date = str(end_date) if end_date else ""

        # Config path
        metadata.config_path = config_path or ""

        return metadata

    def _infer_categories(self, factor_names: List[str]) -> List[str]:
        """
        Infer factor categories from factor names.

        Args:
            factor_names: List of factor names

        Returns:
            List of inferred categories
        """
        categories = set()

        category_keywords = {
            'trend': ['hurst', 'trend', 'emd', 'ma'],
            'volatility': ['vol', 'duvol', 'cv', 'amplitude'],
            'liquidity': ['amihud', 'amivest', 'liquidity'],
            'momentum': ['momentum', 'bias', 'strength'],
            'copula': ['copula'],
            'kalman': ['kalman'],
            'investor_behavior': ['investor', 'behavior'],
            'statistical': ['runs', 'slm', 'skew', 'kurt'],
        }

        for name in factor_names:
            name_lower = name.lower()
            for cat, keywords in category_keywords.items():
                if any(kw in name_lower for kw in keywords):
                    categories.add(cat)
                    break

        return list(categories) if categories else ['mixed']

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert report to DataFrame.

        Returns:
            DataFrame with all report data
        """
        if not self._rows:
            return pd.DataFrame(columns=self.COLUMNS)

        df = pd.DataFrame(self._rows)

        # Reorder columns
        existing_cols = [c for c in self.COLUMNS if c in df.columns]
        df = df[existing_cols]

        return df

    def save(
        self,
        filename: str = "backtest_summary.csv",
        append: bool = True,
        format: str = "csv"
    ) -> str:
        """
        Save report to file.

        Args:
            filename: Output filename (with or without directory)
            append: Whether to append to existing file
            format: Output format ('csv' or 'excel')

        Returns:
            Full path to saved file
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build full path
        if os.path.dirname(filename):
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            filepath = self.output_dir / filename

        # Get current data
        current_df = self.to_dataframe()

        if current_df.empty:
            print("Warning: No data to save")
            return str(filepath)

        # Handle append mode
        if append and filepath.exists():
            try:
                if format == 'csv':
                    existing_df = pd.read_csv(filepath)
                else:
                    existing_df = pd.read_excel(filepath)

                # Combine with existing data
                combined_df = pd.concat([existing_df, current_df], ignore_index=True)
            except Exception as e:
                print(f"Warning: Could not read existing file: {e}")
                combined_df = current_df
        else:
            combined_df = current_df

        # Save to file
        if format == 'csv':
            combined_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        elif format == 'excel':
            if not str(filepath).endswith('.xlsx'):
                filepath = Path(str(filepath).rsplit('.', 1)[0] + '.xlsx')
            combined_df.to_excel(filepath, index=False, sheet_name='Summary')

        print(f"Report saved to: {filepath}")
        return str(filepath)

    def save_both(
        self,
        base_filename: str = "backtest_summary",
        append: bool = True
    ) -> Tuple[str, str]:
        """
        Save report in both CSV and Excel formats.

        Args:
            base_filename: Base filename without extension
            append: Whether to append to existing files

        Returns:
            Tuple of (csv_path, excel_path)
        """
        csv_path = self.save(f"{base_filename}.csv", append=append, format='csv')
        excel_path = self.save(f"{base_filename}.xlsx", append=append, format='excel')
        return csv_path, excel_path

    def clear(self) -> None:
        """Clear all accumulated data."""
        self._rows = []

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of accumulated data.

        Returns:
            Dictionary with summary statistics
        """
        df = self.to_dataframe()

        if df.empty:
            return {}

        # Filter to Total rows only
        total_df = df[df['Year'] == 'Total']

        return {
            'total_backtests': len(total_df),
            'dimensions': total_df['Dimension'].value_counts().to_dict(),
            'backtest_types': total_df['Backtest_Type'].value_counts().to_dict(),
            'avg_sharpe': total_df['Sharpe_Ratio'].mean(),
            'avg_return': total_df['Annualized_Return'].mean(),
            'avg_max_dd': total_df['Max_Drawdown'].mean(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_summary_report(
    equity_curve: pd.Series,
    output_path: str = "Reports/backtest_summary.csv",
    mode: str = None,
    assets: List[str] = None,
    factors: List[str] = None,
    config_path: str = None,
    append: bool = True
) -> str:
    """
    Generate summary report for a single backtest.

    Convenience function for quick report generation.

    Args:
        equity_curve: Daily equity curve
        output_path: Output file path
        mode: Backtest mode string
        assets: Asset list
        factors: Factor names
        config_path: Configuration file path
        append: Whether to append to existing file

    Returns:
        Path to saved report

    Example:
        >>> generate_summary_report(
        ...     equity_curve=result.equity_curve,
        ...     mode="single_factor_single_asset_ts",
        ...     assets=["RB"],
        ...     factors=["HurstExponent"]
        ... )
    """
    report = BacktestSummaryReport()
    report.add_backtest(
        equity_curve=equity_curve,
        mode=mode,
        assets=assets,
        factors=factors,
        config_path=config_path
    )

    # Determine format from extension
    if output_path.endswith('.xlsx'):
        return report.save(output_path, append=append, format='excel')
    else:
        return report.save(output_path, append=append, format='csv')


def create_summary_from_result(
    result: Any,  # BacktestResult
    task_config: Any = None,
    config_path: str = None,
    output_dir: str = "Reports"
) -> 'BacktestSummaryReport':
    """
    Create summary report from BacktestResult object.

    Args:
        result: BacktestResult object
        task_config: TaskConfig object
        config_path: Configuration file path
        output_dir: Output directory

    Returns:
        Populated BacktestSummaryReport object

    Example:
        >>> report = create_summary_from_result(backtest_result, task_config)
        >>> report.save()
    """
    report = BacktestSummaryReport(output_dir=output_dir)
    report.add_from_backtest_result(result, task_config, config_path)
    return report
