#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Time Series Factor Base Class

Base class for single-asset timing factors that compute signals
independently for each asset based on its historical data.

Input: Single asset DataFrame with OHLCV data
Output: float in [-1, 1] representing signal strength
"""

from abc import abstractmethod
from typing import Any, Union
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.factors.base import FactorBase, FactorType


class TimeSeriesFactorBase(FactorBase):
    """
    时序因子基类

    时序因子对单一资产独立计算信号，不依赖其他资产数据。
    典型应用：趋势检测、波动率分析、流动性因子等。

    输入格式:
        DataFrame with columns:
        - TRADE_DT: 日期
        - S_DQ_CLOSE: 收盘价
        - S_DQ_VOLUME: 成交量
        - S_DQ_OI: 持仓量
        - returns: 收益率 (可选，若无则自动计算)

    输出格式:
        float in [-1, 1]
        - 正值表示看多信号
        - 负值表示看空信号
        - 0表示中性信号
    """

    factor_type: FactorType = FactorType.TIME_SERIES

    def __init__(self, name: str, window: int = 20, **kwargs):
        """
        Args:
            name: 因子名称
            window: 回看窗口
            **kwargs: 其他参数
        """
        super().__init__(name=name, window=window)

        # 存储额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)
            self._params[key] = value

    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算时序因子值

        Args:
            data: 单资产历史数据 DataFrame
            **kwargs: 额外参数

        Returns:
            float: 因子信号值 in [-1, 1]
        """
        pass

    def validate_inputs(self, data: pd.DataFrame, **kwargs) -> bool:
        """
        验证输入数据有效性

        Args:
            data: 输入数据

        Returns:
            bool: 数据是否有效
        """
        if data is None or data.empty:
            return False

        if len(data) < self.window:
            return False

        # 检查必需列
        required_cols = ['S_DQ_CLOSE']
        for col in required_cols:
            if col not in data.columns:
                return False

        return True

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据用于计算

        Args:
            data: 原始数据

        Returns:
            DataFrame: 处理后的数据
        """
        df = data.copy()

        # 确保日期排序
        if 'TRADE_DT' in df.columns:
            df = df.sort_values('TRADE_DT')

        # 计算收益率 (如果不存在)
        if 'returns' not in df.columns:
            df['returns'] = df['S_DQ_CLOSE'].pct_change()

        return df

    def _clip_signal(self, signal: float) -> float:
        """
        裁剪信号到 [-1, 1] 范围

        Args:
            signal: 原始信号

        Returns:
            float: 裁剪后的信号
        """
        if pd.isna(signal) or np.isinf(signal):
            return 0.0
        return float(np.clip(signal, -1.0, 1.0))


# Alias for backward compatibility
TimeSeriesFactor = TimeSeriesFactorBase
