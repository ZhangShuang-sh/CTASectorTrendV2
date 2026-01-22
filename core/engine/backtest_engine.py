#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Unified Backtest Engine

Single code path for single-factor and multi-factor backtesting.

Pipeline:
DataLoader -> FactorEngine -> SignalNormalizer -> Combiner -> Executor -> Reporter

Key Features:
- Same code path for single/multi-factor modes (DRY)
- Pluggable components
- Comprehensive logging at every stage
- Support for SIMPLE and PAIR_TRADING modes
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

from core.engine.execution import (
    TradeExecutor,
    CapitalAllocator,
    PortfolioUpdater,
)
from core.engine.execution_engine import (
    ExecutionEngine,
    PairSignal,
    create_execution_engine,
)
from core.engine.risk_manager import RiskManager
from core.engine.position_manager import PositionManager
from core.engine.data_logger import DataLogger


class BacktestMode(Enum):
    """
    Backtest execution mode.

    SIMPLE: Weight-based fast backtesting (factor validation)
    PAIR_TRADING: Full pair trading (margin accounting, risk management, liquidation)
    """
    SIMPLE = "simple"
    PAIR_TRADING = "pair"


@dataclass
class BacktestResult:
    """
    Backtest result data class.

    Attributes:
        portfolio_history: Daily portfolio snapshots
        trade_history: All trade records
        performance_metrics: Performance metrics
        factor_analytics: Factor analysis (IC, IR, etc.)
        daily_logs: Raw log data
        equity_curve: Equity curve series

        # Pair trading mode extra fields
        daily_signals: Daily signal details
        daily_positions: Daily position details
        daily_trades: Daily trade details
        daily_returns: Daily return details
        active_pairs_history: Active pair history
    """
    portfolio_history: List[Dict] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)
    factor_analytics: Dict = field(default_factory=dict)
    daily_logs: Dict = field(default_factory=dict)
    equity_curve: pd.Series = None

    # Pair trading mode extra fields
    daily_signals: List[Dict] = field(default_factory=list)
    daily_positions: List[Dict] = field(default_factory=list)
    daily_trades: List[Dict] = field(default_factory=list)
    daily_returns: List[Dict] = field(default_factory=list)
    active_pairs_history: List[Dict] = field(default_factory=list)


class UnifiedBacktestEngine:
    """
    Unified backtest engine.

    Supports both single-factor and multi-factor backtesting with unified code path.

    Design Principles:
    1. Single/multi-factor use same code path (DRY)
    2. Pluggable components
    3. Comprehensive logging at each stage
    4. Support for SIMPLE and PAIR_TRADING execution modes

    Modes:
    - SIMPLE: Weight-based fast backtesting for factor validation
    - PAIR_TRADING: Full pair trading with margin accounting, risk management

    Pipeline:
    DataLoader -> FactorEngine -> Normalizer -> Combiner -> Executor -> Reporter
    """

    def __init__(
        self,
        config: Dict = None,
        execution_engine: ExecutionEngine = None,
        risk_manager: RiskManager = None,
        position_manager: PositionManager = None,
        data_logger: DataLogger = None,
        mode: BacktestMode = BacktestMode.SIMPLE
    ):
        """
        Initialize backtest engine.

        Args:
            config: Configuration dictionary
            execution_engine: Execution engine (pair trading mode)
            risk_manager: Risk manager (pair trading mode)
            position_manager: Position manager (pair trading mode)
            data_logger: Data logger for comprehensive logging (optional)
            mode: Default execution mode
        """
        self.config = config or {}

        # Backtest configuration
        backtest_config = self.config.get('backtest', {})
        self.initial_capital = backtest_config.get('initial_capital', 1000000)
        self.transaction_cost = backtest_config.get('transaction_cost', 0.001)
        self.slippage = backtest_config.get('slippage', 0.0002)
        self.margin_rate = backtest_config.get('margin_rate', 0.1)
        self.max_position_ratio = backtest_config.get('max_position_ratio', 0.3)
        self.max_leverage = backtest_config.get('max_leverage', 2.0)

        # Default execution mode
        self.default_mode = mode

        # Components
        self.execution_engine = execution_engine
        self.risk_manager = risk_manager
        self.position_manager = position_manager

        # Data logger for comprehensive logging
        logging_config = self.config.get('logging', {})
        if data_logger is not None:
            self.data_logger = data_logger
        else:
            self.data_logger = DataLogger(logging_config)

        # State
        self._portfolio: Dict = {}
        self._portfolio_history: List[Dict] = []
        self._trade_history: List[Dict] = []
        self._equity_curve: Dict[pd.Timestamp, float] = {}

    def run(
        self,
        data_feed: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        pairs: Dict[str, List[Tuple[str, str]]] = None,
        signals: Dict[str, pd.DataFrame] = None,
        backtest_data: pd.DataFrame = None,
        execution_mode: BacktestMode = None,
        enable_risk_management: bool = True,
        enable_liquidation_protection: bool = True,
        liquidation_threshold: float = 0.05,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            data_feed: {ticker: DataFrame} market data
            start_date: Backtest start date
            end_date: Backtest end date
            pairs: {industry: [(asset1, asset2), ...]} pair definitions
            signals: Pre-computed signals (pair trading mode)
            backtest_data: Raw backtest data (for volatility calculation)
            execution_mode: Execution mode (SIMPLE or PAIR_TRADING)
            enable_risk_management: Enable risk management (pair trading mode)
            enable_liquidation_protection: Enable liquidation protection
            liquidation_threshold: Liquidation threshold (default 5%)
            verbose: Print progress

        Returns:
            BacktestResult with logs and metrics
        """
        exec_mode = execution_mode or self.default_mode

        if exec_mode == BacktestMode.PAIR_TRADING:
            return self._run_pair_trading_mode(
                data_feed=data_feed,
                start_date=start_date,
                end_date=end_date,
                pairs=pairs,
                verbose=verbose,
                signals=signals,
                backtest_data=backtest_data,
                enable_risk_management=enable_risk_management,
                enable_liquidation_protection=enable_liquidation_protection,
                liquidation_threshold=liquidation_threshold
            )
        else:
            return self._run_simple_mode(
                data_feed=data_feed,
                start_date=start_date,
                end_date=end_date,
                pairs=pairs,
                verbose=verbose
            )

    def _run_simple_mode(
        self,
        data_feed: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        pairs: Dict[str, List[Tuple[str, str]]] = None,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Simple mode backtest (weight-based fast backtesting).
        """
        self._initialize_portfolio()
        all_dates = self._get_trading_dates(data_feed, start_date, end_date)

        if verbose:
            print(f"Backtest period: {all_dates[0]} to {all_dates[-1]}")
            print(f"Total trading days: {len(all_dates)}")
            print(f"Mode: SIMPLE")

        for i, date in enumerate(all_dates):
            if verbose and i % 50 == 0:
                print(f"Progress: {i}/{len(all_dates)} - {date}")

            daily_data = self._prepare_daily_data(data_feed, date)
            if not daily_data:
                continue

            # Update portfolio value
            self._update_portfolio(daily_data, date)
            self._equity_curve[date] = self._portfolio['total_value']

        # Calculate performance
        equity_series = pd.Series(self._equity_curve)
        performance_metrics = self._calculate_performance_metrics(equity_series)

        if verbose:
            self._print_performance_report(performance_metrics)

        # Get IC metrics from data logger
        factor_analytics = self.data_logger.calculate_ic_metrics() if self.data_logger else {}

        return BacktestResult(
            portfolio_history=self._portfolio_history,
            trade_history=self._trade_history,
            performance_metrics=performance_metrics,
            factor_analytics=factor_analytics,
            daily_logs=self.data_logger.get_all_logs() if self.data_logger else {},
            equity_curve=equity_series
        )

    def _run_pair_trading_mode(
        self,
        data_feed: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        pairs: Dict[str, List[Tuple[str, str]]] = None,
        verbose: bool = True,
        signals: Dict[str, pd.DataFrame] = None,
        backtest_data: pd.DataFrame = None,
        enable_risk_management: bool = True,
        enable_liquidation_protection: bool = True,
        liquidation_threshold: float = 0.05
    ) -> BacktestResult:
        """
        Pair trading mode backtest (full execution with margin accounting).
        """
        # Initialize execution engine
        if self.execution_engine is None:
            self.execution_engine = create_execution_engine(self.config)

        self._initialize_portfolio()

        # Pair trading mode state
        self._daily_signals: List[Dict] = []
        self._daily_positions: List[Dict] = []
        self._daily_trades: List[Dict] = []
        self._daily_returns: List[Dict] = []
        self._active_pairs_history: List[Dict] = []

        all_dates = self._get_trading_dates(data_feed, start_date, end_date)

        if not all_dates:
            if verbose:
                print("Warning: No trading dates")
            return BacktestResult()

        if verbose:
            print(f"Backtest period: {all_dates[0]} to {all_dates[-1]}")
            print(f"Total trading days: {len(all_dates)}")
            print(f"Mode: PAIR_TRADING")

        portfolio_value_history = pd.Series(dtype=float)

        for i, date in enumerate(all_dates):
            if verbose and i % 50 == 0:
                print(f"Progress: {i}/{len(all_dates)} - {date}")

            daily_data = self._prepare_daily_data(data_feed, date)
            if not daily_data:
                continue

            date_prices = self._get_date_prices(daily_data)

            # Liquidation check
            if enable_liquidation_protection:
                if self.execution_engine.check_liquidation(
                    self._portfolio, self.initial_capital
                ):
                    if verbose:
                        print(f"Liquidation triggered: {date}")

                    self._portfolio, liquidation_trades = self.execution_engine.force_liquidate(
                        self._portfolio, date_prices, date
                    )
                    self._trade_history.extend(liquidation_trades)
                    self._daily_trades.extend(liquidation_trades)

                    self._portfolio = self.execution_engine.update_portfolio_value(
                        self._portfolio, date_prices
                    )
                    self._portfolio['date'] = date
                    self._equity_curve[date] = self._portfolio['total_value']
                    self._portfolio_history.append(self._portfolio.copy())
                    continue

            # Risk management check
            if enable_risk_management and self.execution_engine.risk_manager is not None:
                self._portfolio, risk_actions = self.execution_engine.check_risk_limits(
                    self._portfolio, portfolio_value_history, date_prices
                )

                if risk_actions:
                    if verbose:
                        print(f"Risk actions: {date} - {risk_actions}")

                    self._portfolio, risk_trades = self.execution_engine.execute_risk_actions(
                        self._portfolio, risk_actions, date_prices, date
                    )
                    self._trade_history.extend(risk_trades)
                    self._daily_trades.extend(risk_trades)

            # Get day signals
            day_signals = self._get_day_signals(signals, date, pairs)

            # Process signals
            if day_signals:
                self._portfolio, trades = self.execution_engine.process_signals(
                    portfolio=self._portfolio,
                    signals=day_signals,
                    date_prices=date_prices,
                    current_date=date,
                    backtest_data=backtest_data
                )
                self._trade_history.extend(trades)
                self._daily_trades.extend(trades)

            # Update portfolio value
            self._portfolio = self.execution_engine.update_portfolio_value(
                self._portfolio, date_prices
            )
            self._portfolio['date'] = date
            self._equity_curve[date] = self._portfolio['total_value']
            portfolio_value_history[date] = self._portfolio['total_value']

            # Record details
            self._record_daily_details(date, day_signals, date_prices)
            self._portfolio_history.append(self._portfolio.copy())

        # Calculate performance
        equity_series = pd.Series(self._equity_curve)
        performance_metrics = self._calculate_performance_metrics(equity_series)

        if verbose:
            self._print_performance_report(performance_metrics)

        return BacktestResult(
            portfolio_history=self._portfolio_history,
            trade_history=self._trade_history,
            performance_metrics=performance_metrics,
            equity_curve=equity_series,
            daily_signals=self._daily_signals,
            daily_positions=self._daily_positions,
            daily_trades=self._daily_trades,
            daily_returns=self._daily_returns,
            active_pairs_history=self._active_pairs_history
        )

    def _initialize_portfolio(self) -> None:
        """Initialize portfolio state."""
        self._portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'entry_prices': {},
            'entry_dates': {},
            'active_pairs': {},
            'total_value': self.initial_capital,
            'date': None,
            'leverage': 1.0,
            'daily_return': 0.0,
            'cumulative_return': 0.0,
        }
        self._portfolio_history = []
        self._trade_history = []
        self._equity_curve = {}

    def _get_trading_dates(
        self,
        data_feed: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None
    ) -> List[pd.Timestamp]:
        """Get sorted trading dates."""
        all_dates = set()

        for df in data_feed.values():
            if df is not None and not df.empty:
                if isinstance(df.index, pd.DatetimeIndex):
                    all_dates.update(df.index.tolist())

        sorted_dates = sorted(all_dates)

        if start_date is not None:
            sorted_dates = [d for d in sorted_dates if d >= start_date]
        if end_date is not None:
            sorted_dates = [d for d in sorted_dates if d <= end_date]

        return sorted_dates

    def _prepare_daily_data(
        self,
        data_feed: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> Dict[str, pd.DataFrame]:
        """Prepare data up to current date."""
        daily_data = {}

        for ticker, df in data_feed.items():
            if df is None or df.empty:
                continue

            if isinstance(df.index, pd.DatetimeIndex):
                df_until = df[df.index <= date]
            else:
                df_until = df

            if not df_until.empty:
                daily_data[ticker] = df_until

        return daily_data

    def _get_date_prices(
        self,
        daily_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Get current prices from daily data."""
        date_prices = {}

        for ticker, df in daily_data.items():
            if df is None or df.empty:
                continue

            last_row = df.iloc[-1]
            date_prices[ticker] = {
                'close': last_row.get('close', last_row.get('S_DQ_CLOSE', 0)),
                'open': last_row.get('open', last_row.get('S_DQ_OPEN', 0)),
                'high': last_row.get('high', last_row.get('S_DQ_HIGH', 0)),
                'low': last_row.get('low', last_row.get('S_DQ_LOW', 0)),
                'volume': last_row.get('volume', last_row.get('S_DQ_VOLUME', 0)),
            }

        return date_prices

    def _get_day_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        pairs: Dict[str, List[Tuple[str, str]]]
    ) -> List[PairSignal]:
        """Get signals for current date."""
        day_signals = []

        if signals is None:
            return day_signals

        for industry, sig_df in signals.items():
            if sig_df is None or sig_df.empty:
                continue

            # Filter to current date
            if 'date' in sig_df.columns:
                day_df = sig_df[sig_df['date'] == date]
            elif 'TRADE_DT' in sig_df.columns:
                day_df = sig_df[sig_df['TRADE_DT'] == date]
            else:
                continue

            for _, row in day_df.iterrows():
                pair = row.get('pair')
                if pair is None:
                    continue

                # Ensure pair is tuple
                if isinstance(pair, str):
                    pair = tuple(pair.split('_'))
                elif isinstance(pair, list):
                    pair = tuple(pair)

                signal = PairSignal(
                    date=date,
                    pair=pair,
                    signal=int(row.get('signal', 0)),
                    position_strength=float(row.get('position_strength', 1.0)),
                    price1=float(row.get('price1', 0)),
                    price2=float(row.get('price2', 0)),
                    industry=industry,
                    copula_type=str(row.get('copula_type', '')),
                    cond_prob_u=float(row.get('cond_prob_u', 0)),
                    cond_prob_v=float(row.get('cond_prob_v', 0))
                )
                day_signals.append(signal)

        return day_signals

    def _record_daily_details(
        self,
        date: pd.Timestamp,
        day_signals: List,
        date_prices: Dict[str, Dict]
    ) -> None:
        """Record daily details."""
        # Record signals
        for sig in day_signals:
            self._daily_signals.append({
                'date': date,
                'pair': sig.pair,
                'signal': sig.signal,
                'position_strength': sig.position_strength,
                'industry': sig.industry,
                'copula_type': sig.copula_type,
                'cond_prob_u': sig.cond_prob_u,
                'cond_prob_v': sig.cond_prob_v
            })

        # Record positions
        positions = self._portfolio.get('positions', {})
        entry_prices = self._portfolio.get('entry_prices', {})

        for symbol, qty in positions.items():
            if qty == 0:
                continue

            current_price = date_prices.get(symbol, {}).get('close', 0)
            entry_price = entry_prices.get(symbol, current_price)
            unrealized_pnl = (current_price - entry_price) * qty if current_price > 0 else 0

            self._daily_positions.append({
                'date': date,
                'symbol': symbol,
                'quantity': int(round(qty)),
                'entry_price': entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'pct_return': (current_price / entry_price - 1) if entry_price > 0 else 0
            })

        # Record returns
        prev_value = self._portfolio_history[-1]['total_value'] if self._portfolio_history else self.initial_capital
        current_value = self._portfolio.get('total_value', prev_value)
        daily_return = (current_value / prev_value - 1) if prev_value > 0 else 0
        cumulative_return = (current_value / self.initial_capital - 1)

        self._daily_returns.append({
            'date': date,
            'total_value': current_value,
            'cash': self._portfolio.get('cash', 0),
            'margin_used': self._portfolio.get('margin_used', 0),
            'unrealized_pnl': self._portfolio.get('unrealized_pnl', 0),
            'daily_return': daily_return,
            'cumulative_return': cumulative_return,
            'leverage': self._portfolio.get('leverage', 0),
            'gross_exposure': self._portfolio.get('gross_exposure', 0),
            'net_exposure': self._portfolio.get('net_exposure', 0)
        })

        # Record active pairs
        active_pairs = self._portfolio.get('active_pairs', {})
        self._active_pairs_history.append({
            'date': date,
            'active_pairs': list(active_pairs.keys()),
            'count': len(active_pairs)
        })

    def _update_portfolio(
        self,
        data_feed: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> None:
        """Update portfolio value."""
        positions_value = 0.0
        unrealized_pnl = 0.0
        positions = self._portfolio['positions']
        entry_prices = self._portfolio.get('entry_prices', {})

        for ticker, quantity in positions.items():
            df = data_feed.get(ticker)
            if df is None or df.empty:
                continue

            current_price = df['close'].iloc[-1]
            notional_value = abs(quantity) * current_price
            positions_value += notional_value

            entry_price = entry_prices.get(ticker, current_price)
            pnl = (current_price - entry_price) * quantity
            unrealized_pnl += pnl

        prev_value = self._portfolio['total_value']
        self._portfolio['total_value'] = self._portfolio['cash'] + positions_value
        self._portfolio['unrealized_pnl'] = unrealized_pnl
        self._portfolio['date'] = date

        if prev_value > 0:
            self._portfolio['daily_return'] = self._portfolio['total_value'] / prev_value - 1
        self._portfolio['cumulative_return'] = self._portfolio['total_value'] / self.initial_capital - 1

        if self._portfolio['total_value'] > 0:
            self._portfolio['leverage'] = abs(positions_value) / self._portfolio['total_value']

        self._portfolio_history.append(self._portfolio.copy())

        # Log PnL to data logger
        if self.data_logger:
            self.data_logger.log_pnl(date, self._portfolio)
            self.data_logger.log_positions(date, positions, 'actual')

    def _calculate_performance_metrics(
        self,
        equity_curve: pd.Series
    ) -> Dict:
        """Calculate performance metrics."""
        if equity_curve is None or equity_curve.empty:
            return {}

        returns = equity_curve.pct_change().dropna()

        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0
        ann_return = (1 + total_return) ** (252 / len(equity_curve)) - 1 if len(equity_curve) > 1 else 0
        ann_volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = ann_return / ann_volatility if ann_volatility > 0 else 0

        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0

        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'trading_days': len(equity_curve),
            'win_days': (returns > 0).sum(),
            'loss_days': (returns < 0).sum(),
        }

    def _print_performance_report(self, metrics: Dict) -> None:
        """Print performance report."""
        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Total Return:      {metrics.get('total_return', 0) * 100:.2f}%")
        print(f"Annual Return:     {metrics.get('annualized_return', 0) * 100:.2f}%")
        print(f"Annual Volatility: {metrics.get('annualized_volatility', 0) * 100:.2f}%")
        print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio:     {metrics.get('sortino_ratio', 0):.3f}")
        print(f"Max Drawdown:      {metrics.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Calmar Ratio:      {metrics.get('calmar_ratio', 0):.3f}")
        print(f"Trading Days:      {metrics.get('trading_days', 0)}")
        print("=" * 60)

    def export_results(
        self,
        result: BacktestResult,
        output_dir: str = None,
        prefix: str = "backtest"
    ) -> Dict[str, str]:
        """
        Export backtest results.

        Args:
            result: Backtest result
            output_dir: Output directory
            prefix: File name prefix

        Returns:
            {file_type: file_path}
        """
        if output_dir is None:
            output_dir = self.config.get('data', {}).get('output_dir', '../Result')

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        files = {}

        # Export metrics
        metrics_df = pd.DataFrame([result.performance_metrics])
        metrics_path = output_path / f'{prefix}_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_path, index=False)
        files['metrics'] = str(metrics_path)

        # Export equity curve
        if result.equity_curve is not None:
            equity_path = output_path / f'{prefix}_equity_{timestamp}.csv'
            result.equity_curve.to_csv(equity_path)
            files['equity'] = str(equity_path)

        # Export trades
        if result.trade_history:
            trades_df = pd.DataFrame(result.trade_history)
            trades_path = output_path / f'{prefix}_trades_{timestamp}.csv'
            trades_df.to_csv(trades_path, index=False)
            files['trades'] = str(trades_path)

        return files


def run_backtest(
    data_feed: Dict[str, pd.DataFrame],
    config: Dict = None,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    signals: Dict[str, pd.DataFrame] = None,
    execution_mode: BacktestMode = BacktestMode.SIMPLE,
    verbose: bool = True
) -> BacktestResult:
    """
    Convenience function to run backtest.

    Args:
        data_feed: Market data
        config: Configuration dictionary
        start_date: Start date
        end_date: End date
        signals: Pre-computed signals (pair trading mode)
        execution_mode: Execution mode
        verbose: Print progress

    Returns:
        BacktestResult
    """
    engine = UnifiedBacktestEngine(config=config, mode=execution_mode)

    return engine.run(
        data_feed=data_feed,
        start_date=start_date,
        end_date=end_date,
        signals=signals,
        execution_mode=execution_mode,
        verbose=verbose
    )
