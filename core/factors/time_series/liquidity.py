#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Liquidity Time Series Factors

流动性类时序因子

来源: 中泰期货研究所量化CTA因子库

包含因子:
- AMIHUDFactor: 基于Amihud非流动性指标的动量因子
- AmivestFactor: 基于Amivest流动性指标的动量因子

理论背景:
- Amihud (2002): 非流动性指标 = |收益率| / 成交额
- Amivest流动性比率 = 成交额 / |收益率|
- 流动性动量: 短期流动性 vs 长期流动性的比值变化

Source: V1 factors/time_series/liquidity.py (100% logic preserved)
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.factors.time_series.base import TimeSeriesFactorBase
from core.factors.registry import register


@register('AMIHUDFactor')
class AMIHUDFactor(TimeSeriesFactorBase):
    """
    AMIHUD非流动性因子

    计算逻辑:
    1. AMIHUD = return / amount (Amihud非流动性指标)
    2. 使用EWM(指数加权平均)计算短期和长期AMIHUD
    3. Factor = EWM_short / EWM_long

    信号解读:
    - 高值: 短期非流动性上升（可能预示风险上升或趋势转变）
    - 低值: 短期非流动性下降（流动性改善）

    参数:
    - short_period: 短期EWM窗口（默认2）
    - long_period: 长期EWM窗口（默认8，实际使用long+1=9）
    """

    def __init__(
        self,
        name: str = None,
        short_period: int = 2,
        long_period: int = 8,
        window: int = None
    ):
        """
        Args:
            name: 因子名称
            short_period: 短期EWM span参数
            long_period: 长期EWM span参数（实际使用long+1）
            window: 回看窗口（默认自动计算）
        """
        if window is None:
            window = max(long_period * 5, 50)  # 确保足够的预热期

        if name is None:
            name = f"AMIHUD_{short_period}_{long_period}"

        super().__init__(name=name, window=window)

        self.short_period = short_period
        self.long_period = long_period

        self._params.update({
            'short_period': short_period,
            'long_period': long_period
        })

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算AMIHUD因子值

        Args:
            data: 必须包含价格列和成交额/成交量列

        Returns:
            float: 因子值
        """
        if not self.validate_inputs(data):
            return 0.0

        # 支持多种列名格式
        close_col = 'S_DQ_CLOSE' if 'S_DQ_CLOSE' in data.columns else 'close'
        prices = data[close_col]
        returns = prices.pct_change()

        # 获取成交额
        if 'S_DQ_AMOUNT' in data.columns:
            amount = data['S_DQ_AMOUNT']
        elif 'amount' in data.columns:
            amount = data['amount']
        elif 'S_DQ_VOLUME' in data.columns:
            # 用成交量 * 价格近似成交额
            amount = data['S_DQ_VOLUME'] * prices
        elif 'volume' in data.columns:
            amount = data['volume'] * prices
        else:
            return 0.0

        # 计算AMIHUD指标
        amihud = returns / (amount + 1e-10)  # 避免除零

        # EWM计算
        short_ewm = amihud.ewm(span=self.short_period).mean()
        long_ewm = amihud.ewm(span=self.long_period + 1).mean()

        # 计算比值
        factor_value = short_ewm / (long_ewm + 1e-10)

        result = factor_value.iloc[-1]
        if pd.isna(result) or np.isinf(result):
            return 0.0

        # 限制范围
        return float(np.clip(result, -5.0, 5.0))

    def calculate_series(self, data: pd.DataFrame) -> pd.Series:
        """
        计算完整的因子序列

        Args:
            data: 必须包含价格列和成交额列

        Returns:
            pd.Series: 因子值序列
        """
        close_col = 'S_DQ_CLOSE' if 'S_DQ_CLOSE' in data.columns else 'close'
        prices = data[close_col]
        returns = prices.pct_change()

        if 'S_DQ_AMOUNT' in data.columns:
            amount = data['S_DQ_AMOUNT']
        elif 'amount' in data.columns:
            amount = data['amount']
        elif 'S_DQ_VOLUME' in data.columns:
            amount = data['S_DQ_VOLUME'] * prices
        elif 'volume' in data.columns:
            amount = data['volume'] * prices
        else:
            return pd.Series(index=data.index, dtype=float)

        amihud = returns / (amount + 1e-10)

        short_ewm = amihud.ewm(span=self.short_period).mean()
        long_ewm = amihud.ewm(span=self.long_period + 1).mean()

        factor = short_ewm / (long_ewm + 1e-10)

        # 替换无穷值和NaN
        factor = factor.replace([np.inf, -np.inf], np.nan)
        factor = factor.clip(-5.0, 5.0)

        return factor


@register('AmivestFactor')
class AmivestFactor(TimeSeriesFactorBase):
    """
    Amivest流动性因子

    计算逻辑:
    1. abs_return = |close / close.shift() - 1|
    2. sf (短期流动性) = amount.rolling(short).sum() / abs_return.rolling(short).sum()
    3. lf (长期流动性) = amount.rolling(long).sum() / abs_return.rolling(long).sum()
    4. Factor = sf / lf

    信号解读:
    - 高值: 短期流动性相对提升
    - 低值: 短期流动性相对下降

    参数:
    - short_period: 短期滚动窗口（默认12）
    - long_period: 长期滚动窗口（默认32）
    """

    def __init__(
        self,
        name: str = None,
        short_period: int = 12,
        long_period: int = 32,
        window: int = None
    ):
        """
        Args:
            name: 因子名称
            short_period: 短期滚动窗口
            long_period: 长期滚动窗口
            window: 回看窗口（默认自动计算）
        """
        if window is None:
            window = max(long_period * 2, 80)

        if name is None:
            name = f"Amivest_{short_period}_{long_period}"

        super().__init__(name=name, window=window)

        self.short_period = short_period
        self.long_period = long_period

        self._params.update({
            'short_period': short_period,
            'long_period': long_period
        })

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算Amivest因子值

        Args:
            data: 必须包含价格列和成交额列

        Returns:
            float: 因子值
        """
        if not self.validate_inputs(data):
            return 0.0

        close_col = 'S_DQ_CLOSE' if 'S_DQ_CLOSE' in data.columns else 'close'
        prices = data[close_col]

        # 计算绝对收益率
        abs_return = np.abs(prices / prices.shift(1) - 1)

        # 获取成交额
        if 'S_DQ_AMOUNT' in data.columns:
            amount = data['S_DQ_AMOUNT']
        elif 'amount' in data.columns:
            amount = data['amount']
        elif 'S_DQ_VOLUME' in data.columns:
            amount = data['S_DQ_VOLUME'] * prices
        elif 'volume' in data.columns:
            amount = data['volume'] * prices
        else:
            return 0.0

        # 计算短期和长期Amivest
        sf = amount.rolling(self.short_period).sum() / \
             (abs_return.rolling(self.short_period).sum() + 1e-10)
        lf = amount.rolling(self.long_period).sum() / \
             (abs_return.rolling(self.long_period).sum() + 1e-10)

        # 计算比值
        factor_value = sf / (lf + 1e-10)

        result = factor_value.iloc[-1]
        if pd.isna(result) or np.isinf(result):
            return 0.0

        return float(np.clip(result, -5.0, 5.0))

    def calculate_series(self, data: pd.DataFrame) -> pd.Series:
        """
        计算完整的因子序列

        Args:
            data: 必须包含价格列和成交额列

        Returns:
            pd.Series: 因子值序列
        """
        close_col = 'S_DQ_CLOSE' if 'S_DQ_CLOSE' in data.columns else 'close'
        prices = data[close_col]
        abs_return = np.abs(prices / prices.shift(1) - 1)

        if 'S_DQ_AMOUNT' in data.columns:
            amount = data['S_DQ_AMOUNT']
        elif 'amount' in data.columns:
            amount = data['amount']
        elif 'S_DQ_VOLUME' in data.columns:
            amount = data['S_DQ_VOLUME'] * prices
        elif 'volume' in data.columns:
            amount = data['volume'] * prices
        else:
            return pd.Series(index=data.index, dtype=float)

        sf = amount.rolling(self.short_period).sum() / \
             (abs_return.rolling(self.short_period).sum() + 1e-10)
        lf = amount.rolling(self.long_period).sum() / \
             (abs_return.rolling(self.long_period).sum() + 1e-10)

        factor = sf / (lf + 1e-10)

        factor = factor.replace([np.inf, -np.inf], np.nan)
        factor = factor.clip(-5.0, 5.0)

        return factor


# 便捷工厂函数
def create_amihud_factor(
    short_period: int = 2,
    long_period: int = 8,
    **kwargs
) -> AMIHUDFactor:
    """创建AMIHUD因子"""
    return AMIHUDFactor(
        short_period=short_period,
        long_period=long_period,
        **kwargs
    )


def create_amivest_factor(
    short_period: int = 12,
    long_period: int = 32,
    **kwargs
) -> AmivestFactor:
    """创建Amivest因子"""
    return AmivestFactor(
        short_period=short_period,
        long_period=long_period,
        **kwargs
    )
