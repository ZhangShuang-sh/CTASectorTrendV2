#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Position Manager

Volatility-based position sizing:
- Volatility targeting for consistent risk exposure
- Dynamic position adjustment based on market conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class PositionManager:
    """
    Volatility-adjusted position manager.

    Adjusts position sizes to target a specific annualized volatility,
    ensuring consistent risk exposure across different market regimes.
    """

    def __init__(
        self,
        target_volatility: float = 0.15,
        lookback_window: int = 60,
        max_leverage: float = 3.0
    ):
        """
        Initialize position manager.

        Args:
            target_volatility: Target annualized volatility (default 15%)
            lookback_window: Volatility calculation window (default 60 trading days)
            max_leverage: Maximum leverage multiplier (default 3x)
        """
        self.target_volatility = target_volatility
        self.lookback_window = lookback_window
        self.max_leverage = max_leverage
        self.volatility_history: Dict[str, List] = {}

    def calculate_volatility_adjusted_position(
        self,
        data: pd.DataFrame,
        pair: Tuple[str, str],
        current_date: pd.Timestamp,
        base_position: float = 1.0,
        signal_direction: int = 1
    ) -> Tuple[float, Dict]:
        """
        Calculate volatility-adjusted position size.

        Args:
            data: DataFrame with TRADE_DT, PRODUCT_CODE, S_DQ_CLOSE columns
            pair: Tuple of (product1, product2)
            current_date: Current date
            base_position: Base position size (signal strength)
            signal_direction: Signal direction (1: long pair, -1: short pair)

        Returns:
            Tuple of (adjusted_position, volatility_info)
        """
        try:
            pair_data = self._get_pair_data(data, pair, current_date)

            if pair_data is None or len(pair_data) < 20:
                return base_position, {
                    'error': 'Insufficient data',
                    'data_points': len(pair_data) if pair_data is not None else 0
                }

            pair_returns = self._calculate_pair_returns(pair_data, signal_direction)

            if len(pair_returns) < 20:
                return base_position, {
                    'error': 'Insufficient returns data',
                    'returns_count': len(pair_returns)
                }

            historical_vol = self._calculate_volatility(pair_returns)
            self._update_volatility_history(pair, current_date, historical_vol)

            adjustment_factor = self._calculate_adjustment_factor(historical_vol)
            final_leverage = min(adjustment_factor * base_position, self.max_leverage)

            volatility_info = {
                'historical_volatility': historical_vol,
                'adjustment_factor': adjustment_factor,
                'target_volatility': self.target_volatility,
                'final_leverage': final_leverage,
                'returns_count': len(pair_returns),
                'signal_direction': signal_direction,
                'status': 'success'
            }

            return final_leverage, volatility_info

        except Exception as e:
            return base_position, {'error': str(e), 'status': 'error'}

    def _get_pair_data(
        self,
        data: pd.DataFrame,
        pair: Tuple[str, str],
        current_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """Get aligned price data for a pair."""
        try:
            window_data = data[data['TRADE_DT'] <= current_date].tail(self.lookback_window * 2)

            data1 = window_data[window_data['PRODUCT_CODE'] == pair[0]]
            data2 = window_data[window_data['PRODUCT_CODE'] == pair[1]]

            if data1.empty or data2.empty:
                return None

            merged_data = pd.merge(
                data1[['TRADE_DT', 'S_DQ_CLOSE']],
                data2[['TRADE_DT', 'S_DQ_CLOSE']],
                on='TRADE_DT',
                suffixes=('_1', '_2')
            ).sort_values('TRADE_DT')

            if len(merged_data) < 20:
                return None

            return merged_data

        except Exception:
            return None

    def _calculate_pair_returns(
        self,
        pair_data: pd.DataFrame,
        signal_direction: int
    ) -> pd.Series:
        """Calculate pair spread returns."""
        try:
            price_ratio = pair_data['S_DQ_CLOSE_1'] / pair_data['S_DQ_CLOSE_2']
            log_ratio = np.log(price_ratio)
            pair_returns = log_ratio.diff().dropna()

            if signal_direction == -1:
                pair_returns = -pair_returns

            return pair_returns

        except Exception:
            return pd.Series(dtype=float)

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0

        try:
            daily_vol = returns.std()

            if pd.isna(daily_vol) or daily_vol == 0:
                return 0.01

            annual_vol = daily_vol * np.sqrt(252)
            return annual_vol

        except Exception:
            return 0.15

    def _calculate_adjustment_factor(self, historical_vol: float) -> float:
        """Calculate volatility adjustment factor."""
        try:
            if historical_vol <= 0:
                return 1.0

            adjustment_factor = self.target_volatility / historical_vol
            adjustment_factor = max(0.1, min(adjustment_factor, 10.0))

            return adjustment_factor

        except Exception:
            return 1.0

    def _update_volatility_history(
        self,
        pair: Tuple[str, str],
        current_date: pd.Timestamp,
        volatility: float
    ) -> None:
        """Update volatility history for a pair."""
        pair_key = f"{pair[0]}_{pair[1]}"
        if pair_key not in self.volatility_history:
            self.volatility_history[pair_key] = []

        self.volatility_history[pair_key].append({
            'date': current_date,
            'volatility': volatility
        })

        if len(self.volatility_history[pair_key]) > 100:
            self.volatility_history[pair_key] = self.volatility_history[pair_key][-100:]

    def calculate_portfolio_volatility(
        self,
        positions: Dict[str, float],
        volatility_estimates: Dict[str, float],
        correlation_matrix: pd.DataFrame = None
    ) -> float:
        """
        Calculate portfolio-level volatility.

        Args:
            positions: {symbol: position} mapping
            volatility_estimates: {symbol: volatility} mapping
            correlation_matrix: Correlation matrix between symbols

        Returns:
            Annualized portfolio volatility
        """
        if not positions:
            return 0.0

        try:
            symbols = list(positions.keys())

            if correlation_matrix is None:
                total_variance = 0
                for symbol in symbols:
                    vol = volatility_estimates.get(symbol, 0.15)
                    total_variance += (positions[symbol] * vol) ** 2
            else:
                total_variance = 0
                for i, sym1 in enumerate(symbols):
                    vol1 = volatility_estimates.get(sym1, 0.15)
                    for j, sym2 in enumerate(symbols):
                        vol2 = volatility_estimates.get(sym2, 0.15)

                        if sym1 in correlation_matrix.index and sym2 in correlation_matrix.columns:
                            corr = correlation_matrix.loc[sym1, sym2]
                        else:
                            corr = 0

                        total_variance += positions[sym1] * vol1 * positions[sym2] * vol2 * corr

            return np.sqrt(total_variance)

        except Exception:
            return 0.15

    def get_volatility_forecast(
        self,
        pair: Tuple[str, str],
        method: str = 'ewma',
        decay_factor: float = 0.94
    ) -> float:
        """
        Get volatility forecast for a pair.

        Args:
            pair: Tuple of (product1, product2)
            method: Forecast method ('ewma' or 'simple')
            decay_factor: EWMA decay factor

        Returns:
            Forecasted volatility
        """
        try:
            pair_key = f"{pair[0]}_{pair[1]}"
            if pair_key not in self.volatility_history:
                return self.target_volatility

            vol_history = pd.DataFrame(self.volatility_history[pair_key])
            vol_series = vol_history.set_index('date')['volatility']

            if method == 'ewma':
                return vol_series.ewm(alpha=1 - decay_factor).mean().iloc[-1]
            else:
                return vol_series.mean()

        except Exception:
            return self.target_volatility
