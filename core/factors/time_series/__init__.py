#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Time Series Factors

Single-asset timing factors that compute signals independently for each asset.
"""

from core.factors.time_series.base import (
    TimeSeriesFactorBase,
    TimeSeriesFactor,
)

# Trend Factors
from core.factors.time_series.trend import (
    HurstExponent,
    EMDTrend,
)

# Liquidity Factors
from core.factors.time_series.liquidity import (
    AMIHUDFactor,
    AmivestFactor,
)

# Volatility Factors
from core.factors.time_series.volatility import (
    KalmanFilterDeviation,
    KalmanTrendFollower,
)

# Advanced Volatility Factors
from core.factors.time_series.volatility_advanced import (
    DUVOLFactor,
    RunsTestFactor,
)

# Investor Behavior Factors
from core.factors.time_series.investor_behavior import (
    InvestorBehaviorTrendFactor,
    InvestorBehaviorMomentum,
    InvestorBehaviorReversal,
)

# MA Crossover Factors
from core.factors.time_series.ma_crossover import (
    MACrossoverInnovationFactor,
)

# SLM Timing Factors
from core.factors.time_series.slm_timing import (
    SLMTimingFactor,
)

# Price Volume Pattern Factors (DTW-based)
from core.factors.time_series.price_volume_pattern import (
    PriceVolumePatternFactor,
    create_pv_pattern_factor,
    create_price_only_pattern_factor,
)

__all__ = [
    # Base
    'TimeSeriesFactorBase',
    'TimeSeriesFactor',
    # Trend
    'HurstExponent',
    'EMDTrend',
    # Liquidity
    'AMIHUDFactor',
    'AmivestFactor',
    # Volatility
    'KalmanFilterDeviation',
    'KalmanTrendFollower',
    'DUVOLFactor',
    'RunsTestFactor',
    # Investor Behavior
    'InvestorBehaviorTrendFactor',
    'InvestorBehaviorMomentum',
    'InvestorBehaviorReversal',
    # MA Crossover
    'MACrossoverInnovationFactor',
    # SLM Timing
    'SLMTimingFactor',
    # Price Volume Pattern
    'PriceVolumePatternFactor',
    'create_pv_pattern_factor',
    'create_price_only_pattern_factor',
]
