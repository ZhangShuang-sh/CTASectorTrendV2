#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Metrics Module

Performance metrics and data logging:
- PerformanceMetrics: Sharpe, Sortino, Calmar, drawdown
- DataLogger: Comprehensive logging for backtesting
- BacktestSummaryReport: Comprehensive CSV/Excel summary reports
"""

from core.metrics.performance import (
    PerformanceMetrics,
    PerformanceResult,
)

from core.metrics.data_logger import (
    DataLogger,
    DailyLog,
)

from core.metrics.summary_report import (
    BacktestSummaryReport,
    BacktestMetadata,
    YearlyMetrics,
    PerformanceCalculator,
    Dimension,
    BacktestType,
    generate_summary_report,
    create_summary_from_result,
)

__all__ = [
    # Performance
    'PerformanceMetrics',
    'PerformanceResult',
    # Logging
    'DataLogger',
    'DailyLog',
    # Summary Report
    'BacktestSummaryReport',
    'BacktestMetadata',
    'YearlyMetrics',
    'PerformanceCalculator',
    'Dimension',
    'BacktestType',
    'generate_summary_report',
    'create_summary_from_result',
]
