#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Engine Module

Pure execution engine (no factor logic):
- TradeExecutor: Margin-based trade execution
- CapitalAllocator: Multi-pair capital allocation
- PortfolioUpdater: Portfolio value updates
- ExecutionEngine: Unified execution orchestration
- UnifiedBacktestEngine: Backtest execution
- RiskManager: Risk monitoring and limits
- PositionManager: Position sizing
- BacktestVisualizer: Result visualization
"""

# Execution components
from core.engine.execution import (
    TradeExecutor,
    TradeRecord,
    CapitalAllocator,
    PortfolioUpdater,
    filter_valid_entry_signals,
    get_contract_multiplier,
    CONTRACT_MULTIPLIERS,
    reset_trade_verification,
    disable_trade_verification,
    enable_trade_verification,
)

# Execution engine
from core.engine.execution_engine import (
    ExecutionEngine,
    PairSignal,
    create_execution_engine,
)

# Backtest engine
from core.engine.backtest_engine import (
    UnifiedBacktestEngine,
    BacktestResult,
    BacktestMode,
    run_backtest,
)

# Risk management
from core.engine.risk_manager import RiskManager

# Position sizing
from core.engine.position_manager import PositionManager

# Visualization
from core.engine.visualizer import (
    BacktestVisualizer,
    create_visualizer,
)

# Data Logging
from core.engine.data_logger import (
    DataLogger,
    DailyLog,
)

# Unified Runner
from core.engine.backtest_runner import (
    BacktestRunner,
    TaskMode,
    TaskConfig,
    run_backtest as run_task,
    create_task_config,
)

__all__ = [
    # Execution components
    'TradeExecutor',
    'TradeRecord',
    'CapitalAllocator',
    'PortfolioUpdater',
    'filter_valid_entry_signals',
    'get_contract_multiplier',
    'CONTRACT_MULTIPLIERS',
    'reset_trade_verification',
    'disable_trade_verification',
    'enable_trade_verification',
    # Execution engine
    'ExecutionEngine',
    'PairSignal',
    'create_execution_engine',
    # Backtest engine
    'UnifiedBacktestEngine',
    'BacktestResult',
    'BacktestMode',
    'run_backtest',
    # Risk management
    'RiskManager',
    # Position sizing
    'PositionManager',
    # Visualization
    'BacktestVisualizer',
    'create_visualizer',
    # Unified Runner
    'BacktestRunner',
    'TaskMode',
    'TaskConfig',
    'run_task',
    'create_task_config',
    # Data Logging
    'DataLogger',
    'DailyLog',
]
