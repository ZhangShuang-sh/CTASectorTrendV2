#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Risk Manager

Portfolio risk monitoring and control:
- Drawdown monitoring and limits
- VaR calculation
- Concentration limits
- Stop-loss management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class RiskManager:
    """
    Portfolio risk manager.

    Monitors and controls portfolio risk through:
    - Maximum drawdown limits
    - Value-at-Risk (VaR) monitoring
    - Position concentration limits
    - Individual position stop-losses
    """

    def __init__(
        self,
        max_drawdown_limit: float = 0.20,
        var_confidence: float = 0.95,
        concentration_limit: float = 0.3,
        stop_loss: float = 0.05,
        trailing_stop: float = 0.03
    ):
        """
        Initialize risk manager.

        Args:
            max_drawdown_limit: Maximum allowed drawdown (default 20%)
            var_confidence: VaR confidence level (default 95%)
            concentration_limit: Single position concentration limit (default 30%)
            stop_loss: Position stop-loss threshold (default 5%)
            trailing_stop: Trailing stop threshold (default 3%)
        """
        self.max_drawdown_limit = max_drawdown_limit
        self.var_confidence = var_confidence
        self.concentration_limit = concentration_limit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.portfolio_history: List[Dict] = []

    def check_risk_limits(
        self,
        portfolio_value_history: pd.Series,
        current_positions: Dict[str, float],
        current_prices: Dict[str, float],
        entry_prices: Dict[str, float]
    ) -> Tuple[Dict, List[str]]:
        """
        Check all risk limits.

        Args:
            portfolio_value_history: Historical portfolio values
            current_positions: Current positions {symbol: quantity}
            current_prices: Current prices {symbol: price}
            entry_prices: Entry prices {symbol: price}

        Returns:
            Tuple of (risk_status dict, list of required risk actions)
        """
        risk_status = {}
        risk_actions = []

        try:
            # Check drawdown
            max_dd, current_dd = self._calculate_drawdown(portfolio_value_history)
            risk_status['max_drawdown'] = max_dd
            risk_status['current_drawdown'] = current_dd
            risk_status['drawdown_breach'] = max_dd > self.max_drawdown_limit

            if risk_status['drawdown_breach']:
                risk_actions.append('reduce_leverage')

            # Check VaR
            var_breach, var_value = self._check_var_breach(portfolio_value_history)
            risk_status['var_breach'] = var_breach
            risk_status['var_value'] = var_value

            if var_breach:
                risk_actions.append('reduce_risk_exposure')

            # Check concentration
            concentration_breach, concentration_metrics = self._check_concentration(
                current_positions, current_prices
            )
            risk_status['concentration_breach'] = concentration_breach
            risk_status['concentration_metrics'] = concentration_metrics

            if concentration_breach:
                risk_actions.append('diversify_positions')

            # Check stop-losses
            stop_loss_actions = self._check_stop_loss(
                current_positions, current_prices, entry_prices
            )
            risk_actions.extend(stop_loss_actions)
            risk_status['stop_loss_actions'] = stop_loss_actions

            # Overall risk assessment
            risk_status['should_reduce_risk'] = (
                risk_status['drawdown_breach'] or
                risk_status['var_breach'] or
                risk_status['concentration_breach'] or
                len(stop_loss_actions) > 0
            )

            self._record_risk_status(risk_status)

            return risk_status, risk_actions

        except Exception as e:
            return {'error': str(e)}, []

    def _calculate_drawdown(
        self,
        portfolio_values: pd.Series
    ) -> Tuple[float, float]:
        """Calculate maximum and current drawdown."""
        if len(portfolio_values) == 0:
            return 0, 0

        try:
            peak = portfolio_values.expanding().max()
            drawdown = (peak - portfolio_values) / peak

            max_drawdown = drawdown.max()
            current_drawdown = drawdown.iloc[-1] if not pd.isna(drawdown.iloc[-1]) else 0

            return max_drawdown, current_drawdown

        except Exception:
            return 0, 0

    def _check_var_breach(
        self,
        portfolio_values: pd.Series,
        window: int = 60
    ) -> Tuple[bool, float]:
        """Check if VaR limit is breached."""
        if len(portfolio_values) < window:
            return False, 0

        try:
            returns = portfolio_values.pct_change().dropna()

            if len(returns) < window:
                return False, 0

            recent_returns = returns.tail(window)
            var = np.percentile(recent_returns, (1 - self.var_confidence) * 100)

            last_return = returns.iloc[-1]
            breach = last_return < var

            return breach, var

        except Exception:
            return False, 0

    def _check_concentration(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> Tuple[bool, Dict]:
        """Check position concentration risk."""
        if not positions:
            return False, {}

        try:
            position_values = {}
            total_value = 0

            for symbol, position in positions.items():
                if symbol in prices:
                    value = abs(position * prices[symbol])
                    position_values[symbol] = value
                    total_value += value

            if total_value == 0:
                return False, {}

            concentration_metrics = {}
            max_concentration = 0
            most_concentrated = None

            for symbol, value in position_values.items():
                concentration = value / total_value
                concentration_metrics[symbol] = concentration

                if concentration > max_concentration:
                    max_concentration = concentration
                    most_concentrated = symbol

            breach = max_concentration > self.concentration_limit

            metrics = {
                'max_concentration': max_concentration,
                'most_concentrated': most_concentrated,
                'all_concentrations': concentration_metrics,
                'total_value': total_value
            }

            return breach, metrics

        except Exception:
            return False, {}

    def _check_stop_loss(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        entry_prices: Dict[str, float]
    ) -> List[str]:
        """Check individual position stop-losses."""
        actions = []

        try:
            for symbol, position in positions.items():
                if symbol not in current_prices or symbol not in entry_prices:
                    continue

                current_price = current_prices[symbol]
                entry_price = entry_prices[symbol]

                if entry_price == 0:
                    continue

                if position > 0:  # Long position
                    returns = (current_price - entry_price) / entry_price
                    if returns < -self.stop_loss:
                        actions.append(f'stop_loss_{symbol}')

                elif position < 0:  # Short position
                    returns = (entry_price - current_price) / entry_price
                    if returns < -self.stop_loss:
                        actions.append(f'stop_loss_{symbol}')

            return actions

        except Exception:
            return []

    def _record_risk_status(self, risk_status: Dict) -> None:
        """Record risk status history."""
        self.portfolio_history.append({
            'timestamp': pd.Timestamp.now(),
            'risk_status': risk_status
        })

        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]

    def get_risk_report(
        self,
        portfolio_value: float,
        positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> Dict:
        """
        Generate risk report.

        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            prices: Current prices

        Returns:
            Risk report dictionary
        """
        try:
            report = {
                'portfolio_value': portfolio_value,
                'position_count': len(positions),
                'risk_metrics': {}
            }

            if self.portfolio_history:
                recent_risk = self.portfolio_history[-1]['risk_status']
                report['risk_metrics'].update(recent_risk)

            if positions and prices:
                total_long = 0
                total_short = 0

                for symbol, position in positions.items():
                    if symbol in prices:
                        value = position * prices[symbol]
                        if value > 0:
                            total_long += value
                        else:
                            total_short += abs(value)

                report['risk_metrics']['total_long_exposure'] = total_long
                report['risk_metrics']['total_short_exposure'] = total_short
                report['risk_metrics']['net_exposure'] = total_long - total_short
                report['risk_metrics']['gross_exposure'] = total_long + total_short

            return report

        except Exception as e:
            return {'error': str(e)}
