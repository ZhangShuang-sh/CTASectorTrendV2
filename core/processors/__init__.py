#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Processors Module

Signal processing and multi-layer fusion:
- HierarchicalConfigLoader: V2 4-layer configuration parsing
- ParameterResolver: Parameter resolution (idio -> sector -> common)
- SignalNormalizer: Signal normalization (zscore_clip/minmax/rank)
- MultiLayerCombiner: L1-L4 fusion
- FactorComputeEngine: Factor computation orchestration
"""

from core.processors.config_loader import (
    HierarchicalConfigLoader,
    load_config,
)

from core.processors.param_resolver import (
    ParameterResolver,
    create_param_resolver,
)

from core.processors.normalizer import (
    SignalNormalizer,
    NormalizationMethod,
    create_normalizer,
    zscore_normalize,
    rank_normalize,
)

from core.processors.combiner import (
    MultiLayerCombiner,
    CombinedSignal,
    create_combiner,
)

from core.processors.factor_engine import (
    FactorComputeEngine,
    FactorOutput,
    create_factor_engine,
)

__all__ = [
    # Config
    'HierarchicalConfigLoader',
    'load_config',

    # Parameter Resolution
    'ParameterResolver',
    'create_param_resolver',

    # Normalization
    'SignalNormalizer',
    'NormalizationMethod',
    'create_normalizer',
    'zscore_normalize',
    'rank_normalize',

    # Combination
    'MultiLayerCombiner',
    'CombinedSignal',
    'create_combiner',

    # Factor Engine
    'FactorComputeEngine',
    'FactorOutput',
    'create_factor_engine',
]
