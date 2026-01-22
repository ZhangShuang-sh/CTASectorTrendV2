#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Metrics Module

Performance metrics and data logging:
- PerformanceMetrics: Sharpe, Sortino, Calmar, drawdown
- DataLogger: Comprehensive logging for backtesting
"""

from core.metrics.performance import (
    PerformanceMetrics,
    PerformanceResult,
)

from core.metrics.data_logger import (
    DataLogger,
    DailyLog,
)

__all__ = [
    'PerformanceMetrics',
    'PerformanceResult',
    'DataLogger',
    'DailyLog',
]
