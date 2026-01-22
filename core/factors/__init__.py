#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Factors Module

Provides all factor implementations organized by type:
- time_series: Single-asset timing factors
- cross_sectional: Universe ranking factors
- pair_trading: Pairwise comparison factors
"""

from core.factors.base import (
    FactorBase,
    FactorType,
    FactorScope,
    BaseFactor,  # Alias for backward compatibility
)

from core.factors.registry import (
    FactorRegistry,
    get_registry,
    register_factor,
    get_factor,
    register,
)

from core.factors.time_series.base import (
    TimeSeriesFactorBase,
    TimeSeriesFactor,
)

from core.factors.cross_sectional.base import (
    CrossSectionalFactorBase,
    CrossSectionalFactor,
    PairwiseCrossSectionalFactorBase,
    PairwiseCrossSectionalFactor,
)

from core.factors.pair_trading.base import (
    PairTradingFactorBase,
    PairTradingFactor,
    UniversePairTradingFactorBase,
    UniversePairTradingFactor,
)

__all__ = [
    # Base classes
    'FactorBase',
    'FactorType',
    'FactorScope',
    'BaseFactor',

    # Registry
    'FactorRegistry',
    'get_registry',
    'register_factor',
    'get_factor',
    'register',

    # Time series
    'TimeSeriesFactorBase',
    'TimeSeriesFactor',

    # Cross sectional
    'CrossSectionalFactorBase',
    'CrossSectionalFactor',
    'PairwiseCrossSectionalFactorBase',
    'PairwiseCrossSectionalFactor',

    # Pair trading
    'PairTradingFactorBase',
    'PairTradingFactor',
    'UniversePairTradingFactorBase',
    'UniversePairTradingFactor',
]
