#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Execution Engine

Unified execution orchestration for pair trading backtests:
- Signal processing with conflict detection
- Capital allocation (equal/weighted/volatility_parity)
- Margin-based trade execution
- Risk management integration
- Liquidation protection
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from core.engine.execution import (
    TradeExecutor,
    TradeRecord,
    CapitalAllocator,
    PortfolioUpdater,
    filter_valid_entry_signals,
    FALLBACK_MAX_LOTS,
    HARD_MIN_LOTS
)
from core.engine.risk_manager import RiskManager
from core.engine.position_manager import PositionManager


@dataclass
class PairSignal:
    """Pair trading signal data structure."""
    date: pd.Timestamp
    pair: Tuple[str, str]
    signal: int  # 1: long pair, -1: short pair, 0: close
    position_strength: float = 1.0
    price1: float = 0.0
    price2: float = 0.0
    industry: str = ''
    copula_type: str = ''
    cond_prob_u: float = 0.0
    cond_prob_v: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ExecutionEngine:
    """
    Unified execution engine for pair trading.

    Orchestrates all execution components:
    - Signal processing with conflict detection
    - Capital allocation (equal/weighted/volatility_parity)
    - Volatility-targeted position sizing
    - Margin-based trade execution
    - Risk management checks
    - Liquidation protection
    """

    def __init__(
        self,
        trade_executor: TradeExecutor = None,
        capital_allocator: CapitalAllocator = None,
        portfolio_updater: PortfolioUpdater = None,
        risk_manager: RiskManager = None,
        position_manager: PositionManager = None,
        config: Dict = None
    ):
        """
        Initialize execution engine.

        Args:
            trade_executor: Trade execution component
            capital_allocator: Capital allocation component
            portfolio_updater: Portfolio value updater
            risk_manager: Risk management component
            position_manager: Position sizing component
            config: Configuration dictionary
        """
        self.config = config or {}

        backtest_config = self.config.get('backtest', {})
        position_config = self.config.get('position_manager', {})
        risk_config = self.config.get('risk_manager', {})

        # Initialize components
        self.trade_executor = trade_executor or TradeExecutor(
            transaction_cost=backtest_config.get('transaction_cost', 0.001),
            slippage=backtest_config.get('slippage', 0.0002),
            margin_rate=backtest_config.get('margin_rate', 0.1),
            max_position_ratio=backtest_config.get('max_position_ratio', 0.3)
        )

        allocation_method = position_config.get('margin_allocation_method', 'volatility_parity')
        self.capital_allocator = capital_allocator or CapitalAllocator(
            method=allocation_method,
            max_position_ratio=backtest_config.get('max_position_ratio', 0.3),
            default_volatility=position_config.get('target_volatility', 0.15)
        )

        self.portfolio_updater = portfolio_updater or PortfolioUpdater(
            margin_rate=backtest_config.get('margin_rate', 0.1)
        )

        self.risk_manager = risk_manager or RiskManager(
            max_drawdown_limit=risk_config.get('max_drawdown_limit', 0.20),
            var_confidence=risk_config.get('var_confidence', 0.95),
            concentration_limit=risk_config.get('concentration_limit', 0.3),
            stop_loss=risk_config.get('stop_loss', 0.05),
            trailing_stop=risk_config.get('trailing_stop', 0.03)
        )

        self.position_manager = position_manager or PositionManager(
            target_volatility=position_config.get('target_volatility', 0.15),
            lookback_window=position_config.get('lookback_window', 60),
            max_leverage=position_config.get('max_leverage', 3.0)
        )

        # Configuration
        self.liquidation_threshold = backtest_config.get('liquidation_threshold', 0.05)
        self.initial_capital = backtest_config.get('initial_capital', 1000000)

        # Logging
        self._daily_signals: List[Dict] = []
        self._daily_positions: List[Dict] = []
        self._daily_trades: List[Dict] = []
        self._daily_returns: List[Dict] = []

    def process_signals(
        self,
        portfolio: Dict,
        signals: List[PairSignal],
        date_prices: Dict[str, Dict],
        current_date: pd.Timestamp,
        backtest_data: pd.DataFrame = None
    ) -> Tuple[Dict, List[Dict]]:
        """
        Process pair trading signals.

        Pipeline:
        1. Separate entry and exit signals
        2. Process exit signals first (release margin)
        3. Filter valid entry signals (conflict detection)
        4. Allocate capital
        5. Volatility-adjust positions
        6. Execute trades

        Args:
            portfolio: Current portfolio state
            signals: List of PairSignal objects
            date_prices: Current prices {symbol: {'close': price}}
            current_date: Current date
            backtest_data: Historical data for volatility calculation

        Returns:
            Tuple of (updated portfolio, trade list)
        """
        trades = []

        if not signals:
            return portfolio, trades

        # 1. Separate signals
        exit_signals = [s for s in signals if s.signal == 0]
        entry_signals = [s for s in signals if s.signal != 0]

        # 2. Process exit signals
        portfolio, exit_trades = self._process_exit_signals(
            portfolio, exit_signals, date_prices, current_date
        )
        trades.extend(exit_trades)

        # 3. Filter valid entries
        active_pairs = portfolio.get('active_pairs', {})
        valid_entry_signals = self._filter_valid_entries(entry_signals, active_pairs)

        # 4. Allocate capital and execute
        if valid_entry_signals:
            allocations = self.capital_allocator.allocate(
                portfolio=portfolio,
                entry_signals=[(sig.pair, sig) for sig in valid_entry_signals],
                date_prices=date_prices,
                backtest_data=backtest_data,
                current_date=current_date
            )

            # 5. Process each entry signal
            for i, sig in enumerate(valid_entry_signals):
                allocated_capital = allocations[i] if i < len(allocations) else 0

                # Minimum allocation for valid signals
                if allocated_capital <= 0 and abs(sig.signal) > 0.01:
                    allocated_capital = portfolio.get('cash', 0) * 0.01
                    if allocated_capital <= 0:
                        allocated_capital = 100000

                if allocated_capital <= 0:
                    continue

                portfolio, entry_trades = self._process_entry_signal(
                    portfolio=portfolio,
                    signal=sig,
                    allocated_capital=allocated_capital,
                    date_prices=date_prices,
                    current_date=current_date,
                    backtest_data=backtest_data
                )
                trades.extend(entry_trades)

        return portfolio, trades

    def _process_exit_signals(
        self,
        portfolio: Dict,
        exit_signals: List[PairSignal],
        date_prices: Dict[str, Dict],
        current_date: pd.Timestamp
    ) -> Tuple[Dict, List[Dict]]:
        """Process exit (close) signals."""
        trades = []
        active_pairs = portfolio.get('active_pairs', {})
        positions = portfolio.get('positions', {})

        for sig in exit_signals:
            pair = sig.pair
            reverse_pair = (pair[1], pair[0])

            # Find matching active pair
            target_pair = None
            if pair in active_pairs:
                target_pair = pair
            elif reverse_pair in active_pairs:
                target_pair = reverse_pair

            if target_pair is None:
                continue

            symbol1, symbol2 = target_pair
            pos1 = positions.get(symbol1, 0)
            pos2 = positions.get(symbol2, 0)

            # Skip if already closed
            if pos1 == 0 and pos2 == 0:
                del active_pairs[target_pair]
                continue

            # Close both legs
            for symbol in [symbol1, symbol2]:
                current_pos = positions.get(symbol, 0)
                if current_pos == 0:
                    continue

                portfolio, trade = self.trade_executor.execute_trade(
                    portfolio=portfolio,
                    symbol=symbol,
                    target_quantity=0,
                    date_prices=date_prices,
                    reason=f'Close pair {target_pair}',
                    signal=sig
                )
                if trade:
                    trades.append(self._trade_record_to_dict(trade))

            del active_pairs[target_pair]
            self._log_signal(current_date, sig, 'EXIT')

        portfolio['active_pairs'] = active_pairs
        return portfolio, trades

    def _filter_valid_entries(
        self,
        entry_signals: List[PairSignal],
        active_pairs: Dict
    ) -> List[PairSignal]:
        """Filter out conflicting entry signals."""
        valid = []
        used_products = set()

        # Collect products from active pairs
        for pair in active_pairs.keys():
            if isinstance(pair, tuple) and len(pair) >= 2:
                used_products.add(pair[0])
                used_products.add(pair[1])

        for sig in entry_signals:
            pair = sig.pair
            reverse_pair = (pair[1], pair[0])

            # Skip if pair already active
            if pair in active_pairs or reverse_pair in active_pairs:
                continue

            # Skip if products already in use
            if pair[0] in used_products or pair[1] in used_products:
                continue

            valid.append(sig)
            used_products.add(pair[0])
            used_products.add(pair[1])

        return valid

    def _process_entry_signal(
        self,
        portfolio: Dict,
        signal: PairSignal,
        allocated_capital: float,
        date_prices: Dict[str, Dict],
        current_date: pd.Timestamp,
        backtest_data: pd.DataFrame = None
    ) -> Tuple[Dict, List[Dict]]:
        """Process a single entry signal."""
        trades = []
        pair = signal.pair
        direction = signal.signal

        # Volatility adjustment
        vol_adjusted_position = signal.position_strength
        vol_info = {}

        if self.position_manager is not None and backtest_data is not None:
            vol_adjusted_position, vol_info = self.position_manager.calculate_volatility_adjusted_position(
                data=backtest_data,
                pair=pair,
                current_date=current_date,
                base_position=signal.position_strength,
                signal_direction=direction
            )

        # Calculate position quantities
        quantities = self.trade_executor.calculate_position_quantities(
            portfolio=portfolio,
            pair=pair,
            allocated_capital=allocated_capital * vol_adjusted_position,
            date_prices=date_prices,
            backtest_data=backtest_data,
            current_date=current_date
        )

        if quantities is None:
            return portfolio, trades

        qty1, qty2 = quantities

        # Apply limits
        qty1 = min(FALLBACK_MAX_LOTS, max(HARD_MIN_LOTS, qty1))
        qty2 = min(FALLBACK_MAX_LOTS, max(HARD_MIN_LOTS, qty2))

        # Determine target quantities based on direction
        symbol1, symbol2 = pair
        if direction == 1:
            target_qty1, target_qty2 = qty1, -qty2
        else:
            target_qty1, target_qty2 = -qty1, qty2

        # Apply bounds
        target_qty1 = max(-FALLBACK_MAX_LOTS, min(FALLBACK_MAX_LOTS, target_qty1))
        target_qty2 = max(-FALLBACK_MAX_LOTS, min(FALLBACK_MAX_LOTS, target_qty2))

        # Execute leg 1
        portfolio, trade1 = self.trade_executor.execute_trade(
            portfolio=portfolio,
            symbol=symbol1,
            target_quantity=target_qty1,
            date_prices=date_prices,
            reason=f'Open pair {pair} leg1',
            signal=signal,
            vol_info=vol_info
        )
        if trade1:
            trades.append(self._trade_record_to_dict(trade1))

        # Execute leg 2
        portfolio, trade2 = self.trade_executor.execute_trade(
            portfolio=portfolio,
            symbol=symbol2,
            target_quantity=target_qty2,
            date_prices=date_prices,
            reason=f'Open pair {pair} leg2',
            signal=signal,
            vol_info=vol_info
        )
        if trade2:
            trades.append(self._trade_record_to_dict(trade2))

        # Record active pair
        if trade1 or trade2:
            active_pairs = portfolio.get('active_pairs', {})
            active_pairs[pair] = {
                'entry_date': current_date,
                'direction': direction,
                'signal_strength': signal.position_strength,
                'vol_adjusted': vol_adjusted_position,
                'vol_info': vol_info,
                'industry': signal.industry
            }
            portfolio['active_pairs'] = active_pairs
            self._log_signal(current_date, signal, 'ENTRY')

        return portfolio, trades

    def check_risk_limits(
        self,
        portfolio: Dict,
        portfolio_value_history: pd.Series,
        date_prices: Dict[str, Dict]
    ) -> Tuple[Dict, List[str]]:
        """Check risk limits."""
        if self.risk_manager is None:
            return portfolio, []

        current_prices = {
            symbol: data.get('close', 0)
            for symbol, data in date_prices.items()
        }

        risk_status, risk_actions = self.risk_manager.check_risk_limits(
            portfolio_value_history=portfolio_value_history,
            current_positions=portfolio.get('positions', {}),
            current_prices=current_prices,
            entry_prices=portfolio.get('entry_prices', {})
        )

        return portfolio, risk_actions

    def execute_risk_actions(
        self,
        portfolio: Dict,
        risk_actions: List[str],
        date_prices: Dict[str, Dict],
        current_date: pd.Timestamp
    ) -> Tuple[Dict, List[Dict]]:
        """Execute risk management actions."""
        trades = []

        for action in risk_actions:
            if action == 'reduce_leverage':
                portfolio, action_trades = self._reduce_leverage(
                    portfolio, date_prices, current_date, reduction_ratio=0.3
                )
                trades.extend(action_trades)

            elif action == 'reduce_risk_exposure':
                portfolio, action_trades = self._reduce_risk_exposure(
                    portfolio, date_prices, current_date
                )
                trades.extend(action_trades)

            elif action == 'diversify_positions':
                portfolio, action_trades = self._diversify_positions(
                    portfolio, date_prices, current_date
                )
                trades.extend(action_trades)

            elif action.startswith('stop_loss_'):
                symbol = action.replace('stop_loss_', '')
                portfolio, action_trades = self._execute_stop_loss(
                    portfolio, symbol, date_prices, current_date
                )
                trades.extend(action_trades)

        return portfolio, trades

    def _reduce_leverage(
        self,
        portfolio: Dict,
        date_prices: Dict[str, Dict],
        current_date: pd.Timestamp,
        reduction_ratio: float = 0.3
    ) -> Tuple[Dict, List[Dict]]:
        """Reduce leverage by closing largest positions."""
        trades = []
        positions = portfolio.get('positions', {})

        position_values = []
        for symbol, qty in positions.items():
            if qty == 0:
                continue
            price = date_prices.get(symbol, {}).get('close', 0)
            if price > 0:
                value = abs(qty * price)
                position_values.append((symbol, qty, value))

        position_values.sort(key=lambda x: x[2], reverse=True)
        target_reduction = len(position_values) * reduction_ratio

        for i, (symbol, qty, value) in enumerate(position_values):
            if i >= target_reduction:
                break

            portfolio, trade = self.trade_executor.execute_trade(
                portfolio=portfolio,
                symbol=symbol,
                target_quantity=0,
                date_prices=date_prices,
                reason='Risk control - reduce leverage'
            )
            if trade:
                trades.append(self._trade_record_to_dict(trade))

        return portfolio, trades

    def _reduce_risk_exposure(
        self,
        portfolio: Dict,
        date_prices: Dict[str, Dict],
        current_date: pd.Timestamp
    ) -> Tuple[Dict, List[Dict]]:
        """Reduce risk exposure by closing worst-performing positions."""
        trades = []
        positions = portfolio.get('positions', {})
        entry_prices = portfolio.get('entry_prices', {})

        pnl_list = []
        for symbol, qty in positions.items():
            if qty == 0:
                continue
            current_price = date_prices.get(symbol, {}).get('close', 0)
            entry_price = entry_prices.get(symbol, current_price)
            if current_price > 0 and entry_price > 0:
                pnl = (current_price - entry_price) * qty
                pnl_list.append((symbol, pnl))

        pnl_list.sort(key=lambda x: x[1])

        for symbol, pnl in pnl_list[:1]:
            if pnl < 0:
                portfolio, trade = self.trade_executor.execute_trade(
                    portfolio=portfolio,
                    symbol=symbol,
                    target_quantity=0,
                    date_prices=date_prices,
                    reason='Risk control - reduce exposure'
                )
                if trade:
                    trades.append(self._trade_record_to_dict(trade))

        return portfolio, trades

    def _diversify_positions(
        self,
        portfolio: Dict,
        date_prices: Dict[str, Dict],
        current_date: pd.Timestamp
    ) -> Tuple[Dict, List[Dict]]:
        """Diversify by reducing concentrated positions."""
        trades = []
        positions = portfolio.get('positions', {})

        total_value = 0
        position_values = {}

        for symbol, qty in positions.items():
            if qty == 0:
                continue
            price = date_prices.get(symbol, {}).get('close', 0)
            if price > 0:
                value = abs(qty * price)
                position_values[symbol] = value
                total_value += value

        if total_value == 0:
            return portfolio, trades

        concentration_limit = self.capital_allocator.max_position_ratio

        for symbol, value in position_values.items():
            concentration = value / total_value
            if concentration > concentration_limit:
                target_ratio = concentration_limit * 0.9
                current_qty = positions[symbol]
                target_qty = int(current_qty * (target_ratio / concentration))

                portfolio, trade = self.trade_executor.execute_trade(
                    portfolio=portfolio,
                    symbol=symbol,
                    target_quantity=target_qty,
                    date_prices=date_prices,
                    reason='Risk control - diversify'
                )
                if trade:
                    trades.append(self._trade_record_to_dict(trade))

        return portfolio, trades

    def _execute_stop_loss(
        self,
        portfolio: Dict,
        symbol: str,
        date_prices: Dict[str, Dict],
        current_date: pd.Timestamp
    ) -> Tuple[Dict, List[Dict]]:
        """Execute stop-loss for a position."""
        trades = []

        portfolio, trade = self.trade_executor.execute_trade(
            portfolio=portfolio,
            symbol=symbol,
            target_quantity=0,
            date_prices=date_prices,
            reason=f'Stop loss - {symbol}'
        )
        if trade:
            trades.append(self._trade_record_to_dict(trade))

        return portfolio, trades

    def check_liquidation(
        self,
        portfolio: Dict,
        initial_capital: float = None
    ) -> bool:
        """Check if liquidation is required."""
        if initial_capital is None:
            initial_capital = self.initial_capital

        total_value = portfolio.get('total_value', 0)
        threshold_value = initial_capital * self.liquidation_threshold

        return total_value < threshold_value

    def force_liquidate(
        self,
        portfolio: Dict,
        date_prices: Dict[str, Dict],
        current_date: pd.Timestamp
    ) -> Tuple[Dict, List[Dict]]:
        """Force liquidate all positions."""
        trades = []
        positions = portfolio.get('positions', {}).copy()

        for symbol, qty in positions.items():
            if qty == 0:
                continue

            portfolio, trade = self.trade_executor.execute_trade(
                portfolio=portfolio,
                symbol=symbol,
                target_quantity=0,
                date_prices=date_prices,
                reason='Force liquidation'
            )
            if trade:
                trades.append(self._trade_record_to_dict(trade))

        portfolio['active_pairs'] = {}
        return portfolio, trades

    def update_portfolio_value(
        self,
        portfolio: Dict,
        date_prices: Dict[str, Dict]
    ) -> Dict:
        """Update portfolio value."""
        return self.portfolio_updater.update_portfolio_value(portfolio, date_prices)

    def _trade_record_to_dict(self, trade: TradeRecord) -> Dict:
        """Convert TradeRecord to dict."""
        return {
            'date': trade.date,
            'symbol': trade.symbol,
            'action': trade.action,
            'quantity': trade.quantity,
            'price': trade.price,
            'value': trade.value,
            'cost': trade.cost,
            'pnl': trade.pnl,
            'reason': trade.reason,
            'signal_info': trade.signal_info,
            'vol_info': trade.vol_info
        }

    def _log_signal(
        self,
        date: pd.Timestamp,
        signal: PairSignal,
        action_type: str
    ) -> None:
        """Log signal for debugging."""
        self._daily_signals.append({
            'date': date,
            'action_type': action_type,
            'pair': signal.pair,
            'signal': signal.signal,
            'position_strength': signal.position_strength,
            'industry': signal.industry,
            'copula_type': signal.copula_type,
            'cond_prob_u': signal.cond_prob_u,
            'cond_prob_v': signal.cond_prob_v
        })

    def get_daily_logs(self) -> Dict[str, List[Dict]]:
        """Get detailed logs."""
        return {
            'signals': self._daily_signals,
            'positions': self._daily_positions,
            'trades': self._daily_trades,
            'returns': self._daily_returns
        }

    def clear_logs(self) -> None:
        """Clear all logs."""
        self._daily_signals = []
        self._daily_positions = []
        self._daily_trades = []
        self._daily_returns = []


def create_execution_engine(config: Dict = None) -> ExecutionEngine:
    """
    Factory function to create an ExecutionEngine.

    Args:
        config: Configuration dictionary

    Returns:
        ExecutionEngine instance
    """
    return ExecutionEngine(config=config)
