#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Factor Utilities

Math utilities for factor calculations including:
- Copula model tools
- Moving averages
- Statistical normalization
- Volatility calculation
"""

from core.factors.utils.math_utils import (
    CopulaUtils,
    VolatilityCalculator,
    create_volatility_calculator,
    moving_average,
    exponential_moving_average,
    calculate_zscore,
    calculate_returns,
    calculate_volatility,
    calculate_correlation_matrix,
    rank_normalize,
    winsorize,
)

__all__ = [
    'CopulaUtils',
    'VolatilityCalculator',
    'create_volatility_calculator',
    'moving_average',
    'exponential_moving_average',
    'calculate_zscore',
    'calculate_returns',
    'calculate_volatility',
    'calculate_correlation_matrix',
    'rank_normalize',
    'winsorize',
]
