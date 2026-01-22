#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Pair Trading Factors

Pairwise factors for pair trading strategies.
"""

from core.factors.pair_trading.base import (
    PairTradingFactorBase,
    PairTradingFactor,
    UniversePairTradingFactorBase,
    UniversePairTradingFactor,
)

# Copula Factor
from core.factors.pair_trading.copula import (
    CopulaPairFactor,
)

# Trend Following Factor
from core.factors.pair_trading.trend_following import (
    TrendFollowingPairFactor,
)

# Kalman Filter Factor
from core.factors.pair_trading.kalman_filter import (
    KalmanFilterPairFactor,
    KalmanFilter,
    KalmanFilterState,
)

# Style Momentum Factor
from core.factors.pair_trading.style_momentum import (
    StyleMomentumPairFactor,
    EnhancedStyleMomentumPairFactor,
)

__all__ = [
    # Base
    'PairTradingFactorBase',
    'PairTradingFactor',
    'UniversePairTradingFactorBase',
    'UniversePairTradingFactor',
    # Copula
    'CopulaPairFactor',
    # Trend Following
    'TrendFollowingPairFactor',
    # Kalman Filter
    'KalmanFilterPairFactor',
    'KalmanFilter',
    'KalmanFilterState',
    # Style Momentum
    'StyleMomentumPairFactor',
    'EnhancedStyleMomentumPairFactor',
]
