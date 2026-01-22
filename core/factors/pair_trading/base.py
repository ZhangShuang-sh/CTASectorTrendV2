#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Pair Trading Factor Base Classes

Provides base classes for pair trading factors:

1. PairTradingFactorBase (XS_PAIRWISE):
   - Traditional pair trading with fixed pairs
   - Input: (DataFrame, DataFrame) for two assets
   - Output: float in [-1, 1]

2. UniversePairTradingFactorBase (XS_GLOBAL):
   - Scores all possible pairs in a universe
   - Input: Dict[asset, DataFrame]
   - Output: pd.Series with (asset1, asset2) tuple keys
"""

from abc import abstractmethod
from typing import Any, Dict, Tuple, Union
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.factors.base import FactorBase, FactorType


class PairTradingFactorBase(FactorBase):
    """
    配对交易因子基类 (XS_PAIRWISE)

    用于固定配对的交易策略，计算两个资产间的价差/比率信号。
    典型应用：Copula配对、协整价差、Kalman滤波等。

    输入格式:
        (DataFrame, DataFrame)
        两个资产的历史数据

    输出格式:
        float in [-1, 1]
        - 正值: 做多资产1，做空资产2
        - 负值: 做空资产1，做多资产2
        - 0: 无信号或平仓
    """

    factor_type: FactorType = FactorType.XS_PAIRWISE

    # 配对交易因子通常需要较长的回看窗口
    default_normalization: str = "none"  # 配对信号通常不需要额外正则化

    def __init__(
        self,
        name: str,
        window: int = 60,
        entry_threshold: float = 0.5,
        exit_threshold: float = 0.2,
        **kwargs
    ):
        """
        Args:
            name: 因子名称
            window: 回看窗口
            entry_threshold: 开仓阈值
            exit_threshold: 平仓阈值
            **kwargs: 其他参数
        """
        super().__init__(name=name, window=window)

        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self._params['entry_threshold'] = entry_threshold
        self._params['exit_threshold'] = exit_threshold

        for key, value in kwargs.items():
            setattr(self, key, value)
            self._params[key] = value

    @abstractmethod
    def calculate(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        **kwargs
    ) -> float:
        """
        计算配对交易信号

        Args:
            data1: 第一个资产的历史数据
            data2: 第二个资产的历史数据
            **kwargs: 额外参数 (如 current_position)

        Returns:
            float: 配对信号 in [-1, 1]
        """
        pass

    def validate_inputs(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        **kwargs
    ) -> bool:
        """验证输入数据"""
        if data1 is None or data2 is None:
            return False

        if data1.empty or data2.empty:
            return False

        if len(data1) < self.window or len(data2) < self.window:
            return False

        # 检查必需列
        required_cols = ['S_DQ_CLOSE']
        for col in required_cols:
            if col not in data1.columns or col not in data2.columns:
                return False

        return True

    def _align_data(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        对齐两个资产的数据

        Args:
            data1, data2: 两个资产的数据

        Returns:
            Tuple[DataFrame, DataFrame]: 对齐后的数据
        """
        # 确保日期排序
        if 'TRADE_DT' in data1.columns:
            data1 = data1.sort_values('TRADE_DT')
            data2 = data2.sort_values('TRADE_DT')

            # 取共同日期
            common_dates = set(data1['TRADE_DT']) & set(data2['TRADE_DT'])
            data1 = data1[data1['TRADE_DT'].isin(common_dates)]
            data2 = data2[data2['TRADE_DT'].isin(common_dates)]

        return data1, data2

    def _clip_signal(self, signal: float) -> float:
        """裁剪信号到 [-1, 1] 范围"""
        if pd.isna(signal) or np.isinf(signal):
            return 0.0
        return float(np.clip(signal, -1.0, 1.0))


class UniversePairTradingFactorBase(FactorBase):
    """
    全市场配对交易因子基类 (XS_GLOBAL)

    在全市场范围内评估所有可能的配对，返回配对得分。
    用于动态配对选择而非固定配对。

    输入格式:
        Dict[asset_code, DataFrame]
        全市场所有资产的历史数据

    输出格式:
        pd.Series
        - index: (asset1, asset2) 元组
        - values: 配对得分 (越高越好)
    """

    factor_type: FactorType = FactorType.XS_GLOBAL

    def __init__(
        self,
        name: str,
        window: int = 60,
        min_correlation: float = 0.7,
        **kwargs
    ):
        """
        Args:
            name: 因子名称
            window: 回看窗口
            min_correlation: 最小相关性阈值
            **kwargs: 其他参数
        """
        super().__init__(name=name, window=window)

        self.min_correlation = min_correlation
        self._params['min_correlation'] = min_correlation

        for key, value in kwargs.items():
            setattr(self, key, value)
            self._params[key] = value

    @abstractmethod
    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> pd.Series:
        """
        计算全市场配对得分

        Args:
            universe_data: {资产代码: 历史数据DataFrame}
            **kwargs: 额外参数 (如 industry 过滤)

        Returns:
            pd.Series: 配对得分，index为(asset1, asset2)元组
        """
        pass

    def validate_inputs(
        self,
        universe_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> bool:
        """验证输入数据"""
        if universe_data is None or len(universe_data) < 2:
            return False

        valid_count = 0
        for asset, data in universe_data.items():
            if data is not None and not data.empty and len(data) >= self.window:
                valid_count += 1

        # 至少需要2个有效资产
        return valid_count >= 2


# Aliases for backward compatibility
PairTradingFactor = PairTradingFactorBase
UniversePairTradingFactor = UniversePairTradingFactorBase
