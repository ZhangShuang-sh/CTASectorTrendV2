#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Unified Backtest Runner

Orchestrates the complete backtest pipeline based on configuration.

Supports 7 backtest modes:
1. single_factor_single_asset_ts  - 单因子-单资产-时序
2. multi_factor_single_asset_ts   - 多因子-单资产-时序
3. single_factor_multi_asset_xs   - 单因子-多资产-截面
4. multi_factor_multi_asset_xs    - 多因子-多资产-截面
5. single_factor_single_pair      - 单因子-单对资产-配对
6. multi_factor_single_pair       - 多因子-单对资产-配对
7. multi_factor_multi_asset_comprehensive - 多因子-多资产-综合

Pipeline:
Config -> DataLoader -> FactorEngine -> SignalCombiner -> BacktestEngine -> Result
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from core.data import DataLoader, LiquidityScreener, CorrelationCalculator, PairSelector
from core.processors import (
    HierarchicalConfigLoader,
    ParameterResolver,
    SignalNormalizer,
    MultiLayerCombiner,
    FactorComputeEngine,
)
from core.engine.backtest_engine import UnifiedBacktestEngine, BacktestResult, BacktestMode
from core.engine.execution_engine import ExecutionEngine, PairSignal
from core.engine.risk_manager import RiskManager
from core.engine.position_manager import PositionManager
from core.engine.visualizer import BacktestVisualizer
from core.metrics import PerformanceMetrics


class TaskMode(Enum):
    """
    Backtest task mode enumeration.

    Defines 7 standard backtest configurations:
    - Factor dimension: single_factor vs multi_factor
    - Asset dimension: single_asset vs multi_asset vs single_pair vs multi_pair
    - Signal type: ts (time series) vs xs (cross-sectional) vs pair vs comprehensive
    """
    # Time Series Modes
    SINGLE_FACTOR_SINGLE_ASSET_TS = "single_factor_single_asset_ts"
    MULTI_FACTOR_SINGLE_ASSET_TS = "multi_factor_single_asset_ts"

    # Cross-Sectional Modes
    SINGLE_FACTOR_MULTI_ASSET_XS = "single_factor_multi_asset_xs"
    MULTI_FACTOR_MULTI_ASSET_XS = "multi_factor_multi_asset_xs"

    # Pair Trading Modes
    SINGLE_FACTOR_SINGLE_PAIR = "single_factor_single_pair"
    MULTI_FACTOR_SINGLE_PAIR = "multi_factor_single_pair"

    # Comprehensive Mode
    MULTI_FACTOR_MULTI_ASSET_COMPREHENSIVE = "multi_factor_multi_asset_comprehensive"

    @classmethod
    def from_string(cls, mode_str: str) -> 'TaskMode':
        """Parse mode from string."""
        mode_map = {m.value: m for m in cls}
        if mode_str in mode_map:
            return mode_map[mode_str]
        raise ValueError(f"Unknown mode: {mode_str}. Valid modes: {list(mode_map.keys())}")

    def is_single_factor(self) -> bool:
        return 'single_factor' in self.value

    def is_multi_factor(self) -> bool:
        return 'multi_factor' in self.value

    def is_time_series(self) -> bool:
        return self.value.endswith('_ts')

    def is_cross_sectional(self) -> bool:
        return self.value.endswith('_xs')

    def is_pair_trading(self) -> bool:
        return 'pair' in self.value

    def is_comprehensive(self) -> bool:
        return self.value.endswith('_comprehensive')

    def is_single_asset(self) -> bool:
        return 'single_asset' in self.value or 'single_pair' in self.value

    def is_multi_asset(self) -> bool:
        return 'multi_asset' in self.value or 'multi_pair' in self.value


@dataclass
class TaskConfig:
    """Parsed task configuration."""
    name: str
    mode: TaskMode
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    # Assets
    assets: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    use_all_assets: bool = False

    # Pairs
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    auto_select_pairs: bool = False
    pair_selection_params: Dict = field(default_factory=dict)

    # Factors
    factors: Dict[str, List[Dict]] = field(default_factory=dict)
    factor_type_weights: Dict[str, float] = field(default_factory=dict)

    # Position & Risk
    position_config: Dict = field(default_factory=dict)
    risk_config: Dict = field(default_factory=dict)

    # Execution
    execution_config: Dict = field(default_factory=dict)

    # Output
    output_config: Dict = field(default_factory=dict)

    # Liquidity
    liquidity_filter: Dict = field(default_factory=dict)

    # Normalization
    normalization: Dict = field(default_factory=dict)


class BacktestRunner:
    """
    Unified backtest runner.

    Orchestrates the complete backtest pipeline:
    1. Parse task configuration
    2. Load and filter data
    3. Compute factors based on mode
    4. Generate and combine signals
    5. Execute backtest
    6. Generate results and reports

    Usage:
        runner = BacktestRunner()
        result = runner.run_from_config('config/my_task.yaml')

        # Or with inline config
        result = runner.run(task_config)
    """

    def __init__(
        self,
        data_path: str = None,
        industry_info_path: str = None,
        base_config_path: str = None
    ):
        """
        Initialize runner.

        Args:
            data_path: Path to market data parquet file
            industry_info_path: Path to industry info Excel file
            base_config_path: Path to base configuration (multi_factor_config.yaml)
        """
        self.data_path = data_path
        self.industry_info_path = industry_info_path
        self.base_config_path = base_config_path

        # Components (lazy initialization)
        self._data_loader: Optional[DataLoader] = None
        self._liquidity_screener: Optional[LiquidityScreener] = None
        self._pair_selector: Optional[PairSelector] = None
        self._factor_engine: Optional[FactorComputeEngine] = None
        self._backtest_engine: Optional[UnifiedBacktestEngine] = None
        self._visualizer: Optional[BacktestVisualizer] = None

        # Cached data
        self._market_data: Optional[pd.DataFrame] = None
        self._asset_data: Optional[Dict[str, pd.DataFrame]] = None

    def run_from_config(self, config_path: str, verbose: bool = True) -> BacktestResult:
        """
        Run backtest from YAML configuration file.

        Args:
            config_path: Path to task configuration YAML
            verbose: Print progress

        Returns:
            BacktestResult
        """
        task_config = self._parse_config(config_path)
        return self.run(task_config, verbose=verbose)

    def run(self, task_config: TaskConfig, verbose: bool = True) -> BacktestResult:
        """
        Run backtest with parsed configuration.

        Args:
            task_config: Parsed task configuration
            verbose: Print progress

        Returns:
            BacktestResult
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"CTASectorTrendV2 - Backtest Runner")
            print(f"{'='*60}")
            print(f"Task: {task_config.name}")
            print(f"Mode: {task_config.mode.value}")
            print(f"Period: {task_config.start_date} to {task_config.end_date}")
            print(f"{'='*60}\n")

        # Dispatch to appropriate handler based on mode
        mode = task_config.mode

        if mode == TaskMode.SINGLE_FACTOR_SINGLE_ASSET_TS:
            return self._run_single_factor_single_asset_ts(task_config, verbose)

        elif mode == TaskMode.MULTI_FACTOR_SINGLE_ASSET_TS:
            return self._run_multi_factor_single_asset_ts(task_config, verbose)

        elif mode == TaskMode.SINGLE_FACTOR_MULTI_ASSET_XS:
            return self._run_single_factor_multi_asset_xs(task_config, verbose)

        elif mode == TaskMode.MULTI_FACTOR_MULTI_ASSET_XS:
            return self._run_multi_factor_multi_asset_xs(task_config, verbose)

        elif mode == TaskMode.SINGLE_FACTOR_SINGLE_PAIR:
            return self._run_single_factor_single_pair(task_config, verbose)

        elif mode == TaskMode.MULTI_FACTOR_SINGLE_PAIR:
            return self._run_multi_factor_single_pair(task_config, verbose)

        elif mode == TaskMode.MULTI_FACTOR_MULTI_ASSET_COMPREHENSIVE:
            return self._run_comprehensive(task_config, verbose)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # =========================================================================
    # Mode Handlers
    # =========================================================================

    def _run_single_factor_single_asset_ts(
        self,
        config: TaskConfig,
        verbose: bool
    ) -> BacktestResult:
        """Mode 1: 单因子-单资产-时序"""
        if verbose:
            print("[Mode] Single Factor - Single Asset - Time Series")

        # Load data
        asset_data = self._load_asset_data(config)
        if not config.assets:
            raise ValueError("No assets specified for single_asset mode")

        asset = config.assets[0]
        if asset not in asset_data:
            raise ValueError(f"Asset {asset} not found in data")

        # Get single factor
        ts_factors = config.factors.get('time_series', [])
        enabled_factors = [f for f in ts_factors if f.get('enabled', True)]
        if not enabled_factors:
            raise ValueError("No enabled time_series factors")

        factor_config = enabled_factors[0]

        if verbose:
            print(f"Asset: {asset}")
            print(f"Factor: {factor_config['name']}")

        # Initialize factor engine
        engine = self._get_factor_engine()

        # Compute factor for asset
        data = {asset: asset_data[asset]}
        outputs = engine.compute_for_asset(
            asset_data=asset_data[asset],
            asset=asset,
            normalize=True
        )

        # Generate signals
        signals = self._factor_outputs_to_signals(outputs, asset, config)

        # Run backtest
        return self._execute_backtest(
            data_feed=data,
            signals=signals,
            config=config,
            mode=BacktestMode.SIMPLE,
            verbose=verbose
        )

    def _run_multi_factor_single_asset_ts(
        self,
        config: TaskConfig,
        verbose: bool
    ) -> BacktestResult:
        """Mode 2: 多因子-单资产-时序"""
        if verbose:
            print("[Mode] Multi Factor - Single Asset - Time Series")

        # Load data
        asset_data = self._load_asset_data(config)
        if not config.assets:
            raise ValueError("No assets specified")

        asset = config.assets[0]
        if asset not in asset_data:
            raise ValueError(f"Asset {asset} not found")

        if verbose:
            print(f"Asset: {asset}")

        # Initialize components
        engine = self._get_factor_engine()

        # Compute all time series factors
        outputs = engine.compute_for_asset(
            asset_data=asset_data[asset],
            asset=asset,
            normalize=True
        )

        # Combine signals
        combined = engine.combine_signals(outputs, asset=asset)

        if verbose:
            print(f"Combined signal: {combined.signal:.4f}")
            print(f"  TS contribution: {combined.ts_contribution:.4f}")

        # Generate signals
        signals = self._combined_signal_to_signals(combined, asset, config)

        # Run backtest
        data = {asset: asset_data[asset]}
        return self._execute_backtest(
            data_feed=data,
            signals=signals,
            config=config,
            mode=BacktestMode.SIMPLE,
            verbose=verbose
        )

    def _run_single_factor_multi_asset_xs(
        self,
        config: TaskConfig,
        verbose: bool
    ) -> BacktestResult:
        """Mode 3: 单因子-多资产-截面"""
        if verbose:
            print("[Mode] Single Factor - Multi Asset - Cross-Sectional")

        # Load data
        asset_data = self._load_asset_data(config)
        assets = list(asset_data.keys())

        if verbose:
            print(f"Assets: {len(assets)}")

        # Get single cross-sectional factor
        xs_factors = config.factors.get('cross_sectional', [])
        enabled_factors = [f for f in xs_factors if f.get('enabled', True)]

        # Fall back to time_series if no xs factors
        if not enabled_factors:
            ts_factors = config.factors.get('time_series', [])
            enabled_factors = [f for f in ts_factors if f.get('enabled', True)]

        if not enabled_factors:
            raise ValueError("No enabled factors found")

        factor_config = enabled_factors[0]

        if verbose:
            print(f"Factor: {factor_config['name']}")

        # Compute factor for all assets
        engine = self._get_factor_engine()
        outputs = engine.compute_all(
            data=asset_data,
            factor_types=['cross_sectional', 'time_series'],
            normalize=True
        )

        # Generate cross-sectional signals (rank-based)
        signals = self._generate_xs_signals(outputs, assets, config)

        # Run backtest
        return self._execute_backtest(
            data_feed=asset_data,
            signals=signals,
            config=config,
            mode=BacktestMode.SIMPLE,
            verbose=verbose
        )

    def _run_multi_factor_multi_asset_xs(
        self,
        config: TaskConfig,
        verbose: bool
    ) -> BacktestResult:
        """Mode 4: 多因子-多资产-截面"""
        if verbose:
            print("[Mode] Multi Factor - Multi Asset - Cross-Sectional")

        # Load data
        asset_data = self._load_asset_data(config)
        assets = list(asset_data.keys())

        if verbose:
            print(f"Assets: {len(assets)}")

        # Compute all factors
        engine = self._get_factor_engine()

        # Compute for each asset and combine
        all_signals = {}
        for asset in assets:
            if asset not in asset_data:
                continue

            outputs = engine.compute_for_asset(
                asset_data=asset_data[asset],
                asset=asset,
                normalize=True
            )

            combined = engine.combine_signals(outputs, asset=asset)
            all_signals[asset] = combined.signal

        # Cross-sectional ranking
        signals = self._rank_signals_xs(all_signals, config)

        # Run backtest
        return self._execute_backtest(
            data_feed=asset_data,
            signals=signals,
            config=config,
            mode=BacktestMode.SIMPLE,
            verbose=verbose
        )

    def _run_single_factor_single_pair(
        self,
        config: TaskConfig,
        verbose: bool
    ) -> BacktestResult:
        """Mode 5: 单因子-单对资产-配对"""
        if verbose:
            print("[Mode] Single Factor - Single Pair - Pair Trading")

        # Load data
        asset_data = self._load_asset_data(config)

        # Get pair
        if not config.pairs:
            raise ValueError("No pairs specified")

        pair = config.pairs[0]
        asset1, asset2 = pair

        if asset1 not in asset_data or asset2 not in asset_data:
            raise ValueError(f"Pair assets not found: {pair}")

        if verbose:
            print(f"Pair: {asset1} / {asset2}")

        # Get single pair trading factor
        pair_factors = config.factors.get('pair_trading', [])
        enabled_factors = [f for f in pair_factors if f.get('enabled', True)]
        if not enabled_factors:
            raise ValueError("No enabled pair_trading factors")

        factor_config = enabled_factors[0]

        if verbose:
            print(f"Factor: {factor_config['name']}")

        # Compute pair factor
        engine = self._get_factor_engine()
        outputs = engine.compute_for_pair(
            data1=asset_data[asset1],
            data2=asset_data[asset2],
            pair=pair,
            normalize=False  # Pair factors often produce signals directly
        )

        # Generate pair signals
        signals = self._generate_pair_signals(outputs, pair, config)

        # Run backtest with pair trading mode
        return self._execute_backtest(
            data_feed=asset_data,
            signals=signals,
            config=config,
            mode=BacktestMode.PAIR_TRADING,
            pairs={config.sectors[0] if config.sectors else 'default': [pair]},
            verbose=verbose
        )

    def _run_multi_factor_single_pair(
        self,
        config: TaskConfig,
        verbose: bool
    ) -> BacktestResult:
        """Mode 6: 多因子-单对资产-配对"""
        if verbose:
            print("[Mode] Multi Factor - Single Pair - Pair Trading")

        # Load data
        asset_data = self._load_asset_data(config)

        # Get pair
        if not config.pairs:
            raise ValueError("No pairs specified")

        pair = config.pairs[0]
        asset1, asset2 = pair

        if verbose:
            print(f"Pair: {asset1} / {asset2}")

        # Compute all pair factors
        engine = self._get_factor_engine()
        outputs = engine.compute_for_pair(
            data1=asset_data[asset1],
            data2=asset_data[asset2],
            pair=pair,
            normalize=True
        )

        # Combine signals
        combined = engine.combine_signals(outputs)

        if verbose:
            print(f"Combined signal: {combined.signal:.4f}")

        # Generate pair signals
        signals = self._combined_to_pair_signals(combined, pair, config)

        # Run backtest
        return self._execute_backtest(
            data_feed=asset_data,
            signals=signals,
            config=config,
            mode=BacktestMode.PAIR_TRADING,
            pairs={config.sectors[0] if config.sectors else 'default': [pair]},
            verbose=verbose
        )

    def _run_comprehensive(
        self,
        config: TaskConfig,
        verbose: bool
    ) -> BacktestResult:
        """Mode 7: 多因子-多资产-综合"""
        if verbose:
            print("[Mode] Multi Factor - Multi Asset - Comprehensive")

        # Load data
        asset_data = self._load_asset_data(config)
        assets = list(asset_data.keys())

        if verbose:
            print(f"Assets: {len(assets)}")

        # Auto-select pairs if needed
        pairs = config.pairs
        if config.auto_select_pairs:
            pairs = self._auto_select_pairs(asset_data, config)

        if verbose:
            print(f"Pairs: {len(pairs)}")

        # Initialize components
        engine = self._get_factor_engine()

        # Compute all factors for all assets
        all_outputs = {}
        for asset in assets:
            outputs = engine.compute_for_asset(
                asset_data=asset_data[asset],
                asset=asset,
                normalize=True
            )
            all_outputs[asset] = outputs

        # Compute pair factors
        pair_outputs = {}
        for pair in pairs:
            a1, a2 = pair
            if a1 in asset_data and a2 in asset_data:
                outputs = engine.compute_for_pair(
                    data1=asset_data[a1],
                    data2=asset_data[a2],
                    pair=pair,
                    normalize=True
                )
                pair_outputs[pair] = outputs

        # Combine all signals using 4-layer combiner
        combined_signals = {}
        for asset in assets:
            # Merge asset outputs with relevant pair outputs
            merged_outputs = dict(all_outputs.get(asset, {}))

            # Add pair factor outputs where this asset is involved
            for pair, pout in pair_outputs.items():
                if asset in pair:
                    for name, val in pout.items():
                        merged_outputs[f"{name}_{pair[0]}_{pair[1]}"] = val

            combined = engine.combine_signals(merged_outputs, asset=asset)
            combined_signals[asset] = combined

        if verbose:
            print(f"Computed signals for {len(combined_signals)} assets")

        # Generate comprehensive signals
        signals = self._generate_comprehensive_signals(
            combined_signals,
            pair_outputs,
            pairs,
            config
        )

        # Run backtest
        return self._execute_backtest(
            data_feed=asset_data,
            signals=signals,
            config=config,
            mode=BacktestMode.PAIR_TRADING if pairs else BacktestMode.SIMPLE,
            pairs=self._organize_pairs_by_sector(pairs, config) if pairs else None,
            verbose=verbose
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_config(self, config_path: str) -> TaskConfig:
        """Parse YAML configuration file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        # Parse mode
        mode_str = raw.get('mode', 'single_factor_single_asset_ts')
        mode = TaskMode.from_string(mode_str)

        # Parse dates
        date_range = raw.get('date_range', {})
        start_date = pd.Timestamp(date_range.get('start', '2020-01-01'))
        end_date = pd.Timestamp(date_range.get('end', datetime.now().strftime('%Y-%m-%d')))

        # Parse assets
        assets_config = raw.get('assets', {})
        assets = assets_config.get('symbols', [])
        sectors = assets_config.get('sectors', [])
        use_all = assets_config.get('use_all', False)
        liquidity_filter = assets_config.get('liquidity_filter', {})

        # Parse pairs
        pairs_config = raw.get('pairs', {})
        explicit_pairs = pairs_config.get('explicit', [])
        pairs = [tuple(p) for p in explicit_pairs]
        auto_select = pairs_config.get('auto_select', {})
        auto_select_pairs = auto_select.get('enabled', False)

        # Parse factors
        factors = raw.get('factors', {})

        # Parse weights
        factor_type_weights = raw.get('factor_type_weights', {
            'time_series': 0.50,
            'cross_sectional': 0.35,
            'pair_trading': 0.15
        })

        # Parse position/risk/execution
        position_config = raw.get('position', {})
        risk_config = raw.get('risk', {})
        execution_config = raw.get('execution', {})
        output_config = raw.get('output', {})
        normalization = raw.get('normalization', {})

        return TaskConfig(
            name=raw.get('task', {}).get('name', 'unnamed'),
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            assets=assets,
            sectors=sectors,
            use_all_assets=use_all,
            pairs=pairs,
            auto_select_pairs=auto_select_pairs,
            pair_selection_params=auto_select,
            factors=factors,
            factor_type_weights=factor_type_weights,
            position_config=position_config,
            risk_config=risk_config,
            execution_config=execution_config,
            output_config=output_config,
            liquidity_filter=liquidity_filter,
            normalization=normalization
        )

    def _load_asset_data(self, config: TaskConfig) -> Dict[str, pd.DataFrame]:
        """Load and filter asset data."""
        if self._asset_data is not None:
            return self._filter_assets(self._asset_data, config)

        # Load raw data
        loader = self._get_data_loader()
        raw_data, industry_info = loader.load_real_data()

        if raw_data is None:
            raise RuntimeError("Failed to load market data")

        processed = loader.preprocess_wind_data(raw_data, industry_info)

        # Apply liquidity filter
        if config.liquidity_filter.get('enabled', False):
            screener = self._get_liquidity_screener()
            method = config.liquidity_filter.get('method', 'strict')
            valid_assets, processed = screener.screen(processed, method=method)

        # Convert to dict format
        self._asset_data = self._dataframe_to_dict(processed)

        return self._filter_assets(self._asset_data, config)

    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Convert DataFrame to {asset: DataFrame} dict."""
        result = {}

        # Filter to main contracts only
        if 'FS_INFO_TYPE' in df.columns:
            df = df[df['FS_INFO_TYPE'] == '主力合约_Wind调整']

        # Group by PRODUCT_CODE
        if 'PRODUCT_CODE' in df.columns:
            for code, group in df.groupby('PRODUCT_CODE'):
                result[code] = group.sort_values('TRADE_DT').reset_index(drop=True)

        return result

    def _filter_assets(
        self,
        data: Dict[str, pd.DataFrame],
        config: TaskConfig
    ) -> Dict[str, pd.DataFrame]:
        """Filter assets based on config."""
        if config.use_all_assets:
            return data

        result = {}

        # Filter by explicit list
        if config.assets:
            for asset in config.assets:
                if asset in data:
                    result[asset] = data[asset]

        # Filter by sectors
        if config.sectors:
            for asset, df in data.items():
                if 'INDUSTRY' in df.columns:
                    industry = df['INDUSTRY'].iloc[0] if len(df) > 0 else None
                    if industry in config.sectors:
                        result[asset] = df

        # If no filters, return all
        if not result and not config.assets and not config.sectors:
            return data

        return result

    def _get_data_loader(self) -> DataLoader:
        """Get or create data loader."""
        if self._data_loader is None:
            # V2 uses project-local data paths
            project_root = Path(__file__).parent.parent.parent
            default_data_path = project_root / 'data/raw/wind_style_adjusted_futures_contracts.parquet'
            default_industry_path = project_root / 'data/raw/活跃品种基本信息.xlsx'

            self._data_loader = DataLoader(
                data_path=self.data_path or str(default_data_path),
                industry_info_path=self.industry_info_path or str(default_industry_path)
            )
        return self._data_loader

    def _get_liquidity_screener(self) -> LiquidityScreener:
        """Get or create liquidity screener."""
        if self._liquidity_screener is None:
            self._liquidity_screener = LiquidityScreener()
        return self._liquidity_screener

    def _get_factor_engine(self) -> FactorComputeEngine:
        """Get or create factor engine."""
        if self._factor_engine is None:
            config = HierarchicalConfigLoader(self.base_config_path)
            config.load()
            self._factor_engine = FactorComputeEngine(config=config)
        return self._factor_engine

    def _auto_select_pairs(
        self,
        asset_data: Dict[str, pd.DataFrame],
        config: TaskConfig
    ) -> List[Tuple[str, str]]:
        """Auto-select pairs based on correlation."""
        selector = PairSelector()

        # Merge data for correlation
        merged = pd.concat(asset_data.values())

        params = config.pair_selection_params
        method = params.get('method', 'composite')
        min_corr = params.get('min_correlation', 0.6)

        result = selector.select_pairs(
            data=merged,
            method=method,
            min_correlation=min_corr
        )

        pairs = []
        for sector, df in result.items():
            for _, row in df.iterrows():
                if 'product1' in row and 'product2' in row:
                    pairs.append((row['product1'], row['product2']))

        return pairs

    def _organize_pairs_by_sector(
        self,
        pairs: List[Tuple[str, str]],
        config: TaskConfig
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Organize pairs by sector."""
        # For now, put all in one sector
        return {'default': pairs}

    def _factor_outputs_to_signals(
        self,
        outputs: Dict,
        asset: str,
        config: TaskConfig
    ) -> Dict[str, pd.DataFrame]:
        """Convert factor outputs to signal format."""
        # Get the first factor output
        if not outputs:
            return {}

        first_output = list(outputs.values())[0]
        signal_value = first_output.values if hasattr(first_output, 'values') else first_output

        # Create signal DataFrame
        signals = pd.DataFrame({
            'date': [config.end_date],
            'asset': [asset],
            'signal': [signal_value if isinstance(signal_value, (int, float)) else 0.0]
        })

        return {asset: signals}

    def _combined_signal_to_signals(
        self,
        combined,
        asset: str,
        config: TaskConfig
    ) -> Dict[str, pd.DataFrame]:
        """Convert combined signal to signal format."""
        signals = pd.DataFrame({
            'date': [config.end_date],
            'asset': [asset],
            'signal': [combined.signal]
        })
        return {asset: signals}

    def _generate_xs_signals(
        self,
        outputs: Dict,
        assets: List[str],
        config: TaskConfig
    ) -> Dict[str, pd.DataFrame]:
        """Generate cross-sectional ranked signals."""
        # Extract values for all assets
        values = {}
        for name, output in outputs.items():
            if hasattr(output, 'values'):
                if isinstance(output.values, dict):
                    for asset, val in output.values.items():
                        if asset not in values:
                            values[asset] = []
                        values[asset].append(val)

        # Average and rank
        avg_values = {a: np.mean(v) for a, v in values.items() if v}

        if not avg_values:
            return {}

        # Rank to [-1, 1]
        sorted_assets = sorted(avg_values.keys(), key=lambda x: avg_values[x])
        n = len(sorted_assets)
        ranks = {a: (i / (n - 1) * 2 - 1) if n > 1 else 0 for i, a in enumerate(sorted_assets)}

        result = {}
        for asset in assets:
            if asset in ranks:
                result[asset] = pd.DataFrame({
                    'date': [config.end_date],
                    'asset': [asset],
                    'signal': [ranks[asset]]
                })

        return result

    def _rank_signals_xs(
        self,
        signals: Dict[str, float],
        config: TaskConfig
    ) -> Dict[str, pd.DataFrame]:
        """Rank signals cross-sectionally."""
        if not signals:
            return {}

        sorted_assets = sorted(signals.keys(), key=lambda x: signals[x])
        n = len(sorted_assets)
        ranks = {a: (i / (n - 1) * 2 - 1) if n > 1 else 0 for i, a in enumerate(sorted_assets)}

        result = {}
        for asset, rank in ranks.items():
            result[asset] = pd.DataFrame({
                'date': [config.end_date],
                'asset': [asset],
                'signal': [rank]
            })

        return result

    def _generate_pair_signals(
        self,
        outputs: Dict,
        pair: Tuple[str, str],
        config: TaskConfig
    ) -> Dict:
        """Generate pair trading signals."""
        # Get signal from first factor output
        if not outputs:
            return {}

        first_output = list(outputs.values())[0]
        signal_value = first_output.values if hasattr(first_output, 'values') else first_output

        if isinstance(signal_value, (int, float)):
            signal = 1 if signal_value > 0 else (-1 if signal_value < 0 else 0)
        else:
            signal = 0

        return {
            'default': pd.DataFrame({
                'date': [config.end_date],
                'pair': [pair],
                'signal': [signal],
                'position_strength': [1.0]
            })
        }

    def _combined_to_pair_signals(
        self,
        combined,
        pair: Tuple[str, str],
        config: TaskConfig
    ) -> Dict:
        """Convert combined signal to pair signals."""
        signal = 1 if combined.signal > 0 else (-1 if combined.signal < 0 else 0)

        return {
            'default': pd.DataFrame({
                'date': [config.end_date],
                'pair': [pair],
                'signal': [signal],
                'position_strength': [abs(combined.signal)]
            })
        }

    def _generate_comprehensive_signals(
        self,
        combined_signals: Dict,
        pair_outputs: Dict,
        pairs: List[Tuple[str, str]],
        config: TaskConfig
    ) -> Dict:
        """Generate comprehensive signals combining all types."""
        result = {}

        # Asset-level signals
        for asset, combined in combined_signals.items():
            result[asset] = pd.DataFrame({
                'date': [config.end_date],
                'asset': [asset],
                'signal': [combined.signal]
            })

        # Pair signals
        if pairs:
            pair_signals = []
            for pair in pairs:
                if pair in pair_outputs:
                    outputs = pair_outputs[pair]
                    if outputs:
                        first_output = list(outputs.values())[0]
                        val = first_output.values if hasattr(first_output, 'values') else first_output
                        signal = 1 if val > 0 else (-1 if val < 0 else 0)
                        pair_signals.append({
                            'date': config.end_date,
                            'pair': pair,
                            'signal': signal,
                            'position_strength': 1.0
                        })

            if pair_signals:
                result['_pairs'] = pd.DataFrame(pair_signals)

        return result

    def _execute_backtest(
        self,
        data_feed: Dict[str, pd.DataFrame],
        signals: Dict,
        config: TaskConfig,
        mode: BacktestMode,
        pairs: Dict = None,
        verbose: bool = True
    ) -> BacktestResult:
        """Execute backtest with configured engine."""
        # Build config dict
        engine_config = {
            'backtest': {
                'initial_capital': config.execution_config.get('initial_capital', 10000000),
                'transaction_cost': config.execution_config.get('transaction_cost', 0.001),
                'slippage': config.execution_config.get('slippage', 0.0002),
                'margin_rate': config.execution_config.get('margin_rate', 0.1),
                'max_position_ratio': config.position_config.get('max_single_position', 0.3),
                'max_leverage': config.position_config.get('max_leverage', 3.0),
            }
        }

        # Create components for pair trading mode
        risk_manager = None
        position_manager = None
        execution_engine = None

        if mode == BacktestMode.PAIR_TRADING:
            risk_manager = RiskManager(
                max_drawdown=config.risk_config.get('max_drawdown', 0.15),
                max_single_position=config.risk_config.get('concentration_limit', 0.25),
                stop_loss_threshold=config.risk_config.get('stop_loss_threshold', 0.05)
            )
            position_manager = PositionManager(
                target_volatility=config.position_config.get('target_volatility', 0.15),
                max_leverage=config.position_config.get('max_leverage', 3.0)
            )
            execution_engine = ExecutionEngine(
                risk_manager=risk_manager,
                position_manager=position_manager
            )

        # Create backtest engine
        engine = UnifiedBacktestEngine(
            config=engine_config,
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            position_manager=position_manager,
            mode=mode
        )

        # Run backtest
        result = engine.run(
            data_feed=data_feed,
            start_date=config.start_date,
            end_date=config.end_date,
            pairs=pairs,
            signals=signals.get('_pairs') if '_pairs' in signals else None,
            verbose=verbose
        )

        # Generate visualizations if configured
        if config.output_config.get('generate_plots', False):
            self._generate_plots(result, config)

        return result

    def _generate_plots(self, result: BacktestResult, config: TaskConfig) -> None:
        """Generate visualization plots."""
        if self._visualizer is None:
            self._visualizer = BacktestVisualizer()

        output_dir = config.output_config.get('result_dir', './Result')
        self._visualizer.save_all_plots(
            result=result,
            output_dir=output_dir,
            prefix=config.name
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def run_backtest(config_path: str, verbose: bool = True) -> BacktestResult:
    """
    Run backtest from configuration file.

    Args:
        config_path: Path to task configuration YAML
        verbose: Print progress

    Returns:
        BacktestResult
    """
    runner = BacktestRunner()
    return runner.run_from_config(config_path, verbose=verbose)


def create_task_config(
    mode: str,
    assets: List[str] = None,
    pairs: List[Tuple[str, str]] = None,
    factors: Dict = None,
    **kwargs
) -> TaskConfig:
    """
    Create task configuration programmatically.

    Args:
        mode: Task mode string
        assets: Asset list
        pairs: Pair list
        factors: Factor configuration
        **kwargs: Additional parameters

    Returns:
        TaskConfig
    """
    return TaskConfig(
        name=kwargs.get('name', 'programmatic_task'),
        mode=TaskMode.from_string(mode),
        start_date=pd.Timestamp(kwargs.get('start_date', '2020-01-01')),
        end_date=pd.Timestamp(kwargs.get('end_date', datetime.now().strftime('%Y-%m-%d'))),
        assets=assets or [],
        pairs=pairs or [],
        factors=factors or {},
        factor_type_weights=kwargs.get('factor_type_weights', {
            'time_series': 0.50,
            'cross_sectional': 0.35,
            'pair_trading': 0.15
        }),
        position_config=kwargs.get('position_config', {}),
        risk_config=kwargs.get('risk_config', {}),
        execution_config=kwargs.get('execution_config', {}),
        output_config=kwargs.get('output_config', {})
    )
