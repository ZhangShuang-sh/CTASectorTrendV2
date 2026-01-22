#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Cross-Sectional Factors

Universe ranking factors and pairwise comparison factors.
"""

from core.factors.cross_sectional.base import (
    CrossSectionalFactorBase,
    CrossSectionalFactor,
    PairwiseCrossSectionalFactorBase,
    PairwiseCrossSectionalFactor,
)

# Momentum Factors
from core.factors.cross_sectional.momentum import (
    MomentumRank,
    TermStructure,
)

# Fundamental Factors
from core.factors.cross_sectional.fundamental import (
    MemberHoldings,
)

# CITIC Momentum Factors (5 factors)
from core.factors.cross_sectional.momentum_citic import (
    MomentumDirection,
    CrossSectionalMomentumFactor,
    TimeSeriesMomentumFactor,
    CompositeMomentumFactor,
    BIASFactor,
    TrendStrengthFactor,
    create_momentum_factors,
    CSMomentum,
    TSMomentum,
    ComboMomentum,
    BIAS,
    TrendStr,
)

# Price Volume Factors (6 factors)
from core.factors.cross_sectional.price_volume import (
    FactorDirection,
    VolatilityFactor,
    CVFactor,
    SkewnessFactor,
    KurtosisFactor,
    AmplitudeFactor,
    LiquidityFactor,
    create_price_volume_factors,
)

# VCRR Factors
from core.factors.cross_sectional.vcrr import (
    VCRRFactor,
    VCRRTimeSeriesFactor,
    create_vcrr_factor,
    create_vcrr_ts_factor,
)

__all__ = [
    # Base
    'CrossSectionalFactorBase',
    'CrossSectionalFactor',
    'PairwiseCrossSectionalFactorBase',
    'PairwiseCrossSectionalFactor',
    # Momentum
    'MomentumRank',
    'TermStructure',
    # Fundamental
    'MemberHoldings',
    # CITIC Momentum
    'MomentumDirection',
    'CrossSectionalMomentumFactor',
    'TimeSeriesMomentumFactor',
    'CompositeMomentumFactor',
    'BIASFactor',
    'TrendStrengthFactor',
    'create_momentum_factors',
    'CSMomentum',
    'TSMomentum',
    'ComboMomentum',
    'BIAS',
    'TrendStr',
    # Price Volume
    'FactorDirection',
    'VolatilityFactor',
    'CVFactor',
    'SkewnessFactor',
    'KurtosisFactor',
    'AmplitudeFactor',
    'LiquidityFactor',
    'create_price_volume_factors',
    # VCRR
    'VCRRFactor',
    'VCRRTimeSeriesFactor',
    'create_vcrr_factor',
    'create_vcrr_ts_factor',
]
