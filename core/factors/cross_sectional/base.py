#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Cross-Sectional Factor Base Classes

Provides base classes for two types of cross-sectional factors:

1. CrossSectionalFactorBase (XS_GLOBAL):
   - Ranks N assets in a universe
   - Input: Dict[asset, DataFrame]
   - Output: pd.Series with asset keys and normalized ranks

2. PairwiseCrossSectionalFactorBase (XS_PAIRWISE):
   - Compares exactly 2 assets
   - Input: (DataFrame, DataFrame) for two assets
   - Output: float in [-1, 1]
"""

from abc import abstractmethod
from typing import Any, Dict, Union
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.factors.base import FactorBase, FactorType


class CrossSectionalFactorBase(FactorBase):
    """
    全局截面因子基类 (XS_GLOBAL)

    在N个资产间进行排名或评分。
    典型应用：动量排名、波动率排名、流动性排名等。

    输入格式:
        Dict[asset_code, DataFrame]
        每个DataFrame包含单个资产的历史数据

    输出格式:
        pd.Series
        - index: 资产代码
        - values: 归一化后的因子值 (通常为排名分位数 [0, 1] 或 z-score)
    """

    factor_type: FactorType = FactorType.XS_GLOBAL

    def __init__(self, name: str, window: int = 20, **kwargs):
        """
        Args:
            name: 因子名称
            window: 回看窗口
            **kwargs: 其他参数
        """
        super().__init__(name=name, window=window)

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
        计算全局截面因子值

        Args:
            universe_data: {资产代码: 历史数据DataFrame}
            **kwargs: 额外参数

        Returns:
            pd.Series: 各资产的因子值
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

        for asset, data in universe_data.items():
            if data is None or data.empty:
                continue
            if len(data) < self.window:
                continue
            # At least one valid asset
            return True

        return False

    def _rank_normalize(self, series: pd.Series) -> pd.Series:
        """
        排名归一化到 [0, 1]

        Args:
            series: 原始因子值

        Returns:
            pd.Series: 排名分位数
        """
        return series.rank(pct=True)

    def _zscore_normalize(self, series: pd.Series) -> pd.Series:
        """
        Z-Score 标准化

        Args:
            series: 原始因子值

        Returns:
            pd.Series: 标准化后的因子值
        """
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std


class PairwiseCrossSectionalFactorBase(FactorBase):
    """
    配对截面因子基类 (XS_PAIRWISE)

    比较特定的2个资产，生成配对交易信号。
    典型应用：Copula配对、Kalman滤波价差、风格动量配对等。

    输入格式:
        (DataFrame, DataFrame)
        两个资产的历史数据

    输出格式:
        float in [-1, 1]
        - 正值表示做多第一个资产，做空第二个资产
        - 负值表示做空第一个资产，做多第二个资产
        - 0表示无信号
    """

    factor_type: FactorType = FactorType.XS_PAIRWISE

    def __init__(self, name: str, window: int = 60, **kwargs):
        """
        Args:
            name: 因子名称
            window: 回看窗口 (配对因子通常需要较长窗口)
            **kwargs: 其他参数
        """
        super().__init__(name=name, window=window)

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
        计算配对因子值

        Args:
            data1: 第一个资产的历史数据
            data2: 第二个资产的历史数据
            **kwargs: 额外参数

        Returns:
            float: 配对信号值 in [-1, 1]
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

        return True

    def _clip_signal(self, signal: float) -> float:
        """裁剪信号到 [-1, 1] 范围"""
        if pd.isna(signal) or np.isinf(signal):
            return 0.0
        return float(np.clip(signal, -1.0, 1.0))


# Aliases for backward compatibility
CrossSectionalFactor = CrossSectionalFactorBase
PairwiseCrossSectionalFactor = PairwiseCrossSectionalFactorBase
