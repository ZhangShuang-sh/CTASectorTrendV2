#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进均线交叉因子

基于广发证券《均线交叉策略的另类创新研究》（另类交易策略系列之二十七）

核心创新：
- 传统策略问题：金叉做多、死叉做空，但表现不稳定
- 改进策略：改变平仓条件，引入浮动止盈机制

改进后的平仓规则：
- 多单：当前价格 < 最新金叉价格 → 平仓
- 空单：当前价格 > 最新死叉价格 → 平仓
- 参考价格更新：出现新的同向信号时，更新参考价格

策略特点：
- 低胜率（~13%）但高盈亏比（~9）
- 错误开仓快速止损，正确开仓最大化利润
- 支持多种均线类型：SMA, EMA, WMA, LLT
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from enum import Enum

from core.factors.time_series.base import TimeSeriesFactorBase


class MAType(Enum):
    """均线类型枚举"""
    SMA = 'SMA'  # 简单移动平均
    EMA = 'EMA'  # 指数移动平均
    WMA = 'WMA'  # 加权移动平均
    LLT = 'LLT'  # 低延迟趋势线


class MACrossoverInnovationFactor(TimeSeriesFactorBase):
    """
    改进均线交叉因子

    核心创新：
    1. 开仓：金叉做多，死叉做空
    2. 平仓：基于价格比较的浮动止盈机制
       - 多单：当前价 < 最新金叉价 → 平仓
       - 空单：当前价 > 最新死叉价 → 平仓
    3. 参考价格更新：新同向信号出现时更新参考价格

    信号输出：
    - +1: 持有多仓
    - -1: 持有空仓
    - 0: 空仓
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        ma_type: str = 'WMA',
        name: str = None
    ):
        """
        Args:
            fast_period: 快线周期
            slow_period: 慢线周期
            ma_type: 均线类型 ('SMA', 'EMA', 'WMA', 'LLT')
            name: 因子名称
        """
        if name is None:
            name = f"MACrossoverInnovation_{ma_type}_{fast_period}_{slow_period}"

        super().__init__(
            name=name,
            window=max(slow_period * 2, 60)
        )

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type.upper()

        # 状态变量（用于逐步计算）
        self._position = 0  # 当前持仓方向 1/-1/0
        self._reference_price = None  # 参考价格（最新交叉价格）

    def set_params(self, **kwargs) -> 'MACrossoverInnovationFactor':
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    # ========== 均线计算方法 ==========

    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """简单移动平均"""
        return prices.rolling(window=period).mean()

    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """指数移动平均"""
        alpha = 2.0 / (period + 1)
        return prices.ewm(alpha=alpha, adjust=False).mean()

    def _calculate_wma(self, prices: pd.Series, period: int) -> pd.Series:
        """加权移动平均"""
        weights = np.arange(1, period + 1, dtype=float)

        def weighted_avg(x):
            if len(x) < period:
                return np.nan
            return np.sum(x[-period:] * weights) / weights.sum()

        return prices.rolling(window=period).apply(weighted_avg, raw=True)

    def _calculate_llt(self, prices: pd.Series, period: int) -> pd.Series:
        """低延迟趋势线"""
        alpha = 2.0 / (period + 1)
        n = len(prices)
        llt = np.zeros(n)
        p = prices.values

        for t in range(n):
            if t < 2:
                llt[t] = p[t]
            else:
                llt[t] = ((alpha - alpha ** 2 / 4) * p[t]
                          + (alpha ** 2 / 2) * p[t - 1]
                          - (alpha - 3 * alpha ** 2 / 4) * p[t - 2]
                          + 2 * (1 - alpha) * llt[t - 1]
                          - (1 - alpha) ** 2 * llt[t - 2])

        return pd.Series(llt, index=prices.index)

    def _calculate_ma(self, prices: pd.Series, period: int) -> pd.Series:
        """计算均线"""
        if self.ma_type == 'SMA':
            return self._calculate_sma(prices, period)
        elif self.ma_type == 'EMA':
            return self._calculate_ema(prices, period)
        elif self.ma_type == 'WMA':
            return self._calculate_wma(prices, period)
        elif self.ma_type == 'LLT':
            return self._calculate_llt(prices, period)
        else:
            raise ValueError(f"Unknown MA type: {self.ma_type}")

    # ========== 核心策略逻辑 ==========

    def calculate(self, data: pd.DataFrame) -> float:
        """
        计算当前信号

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            float: 信号值 (-1, 0, 1)
        """
        signals = self.compute_series(data)
        if len(signals) == 0:
            return 0.0
        return float(signals.iloc[-1])

    def compute_series(self, data: pd.DataFrame) -> pd.Series:
        """
        计算完整信号序列

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            pd.Series: 信号序列 (-1, 0, 1)
        """
        prices = self._get_close_prices(data)
        if prices is None or len(prices) < self.slow_period + 1:
            return pd.Series(dtype=float)

        # 计算快慢均线
        fast_ma = self._calculate_ma(prices, self.fast_period)
        slow_ma = self._calculate_ma(prices, self.slow_period)

        # 计算传统交叉信号
        # 金叉：fast_ma > slow_ma 且前一根 fast_ma <= slow_ma
        # 死叉：fast_ma < slow_ma 且前一根 fast_ma >= slow_ma
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        # 改进策略：逐日计算持仓和信号
        n = len(prices)
        signals = pd.Series(index=prices.index, dtype=float)
        signals[:] = 0

        position = 0  # 当前持仓方向
        reference_price = None  # 参考价格

        for i in range(self.slow_period, n):
            current_price = prices.iloc[i]

            # 检查是否有交叉信号
            if golden_cross.iloc[i]:
                # 金叉信号
                if position == 1:
                    # 已持有多仓，更新参考价格
                    reference_price = current_price
                else:
                    # 开多仓
                    position = 1
                    reference_price = current_price

            elif death_cross.iloc[i]:
                # 死叉信号
                if position == -1:
                    # 已持有空仓，更新参考价格
                    reference_price = current_price
                else:
                    # 开空仓
                    position = -1
                    reference_price = current_price

            # 检查平仓条件（改进策略的核心）
            if position == 1 and reference_price is not None:
                # 持有多仓：当前价 < 参考价 → 平仓
                if current_price < reference_price:
                    position = 0
                    reference_price = None

            elif position == -1 and reference_price is not None:
                # 持有空仓：当前价 > 参考价 → 平仓
                if current_price > reference_price:
                    position = 0
                    reference_price = None

            signals.iloc[i] = position

        return signals

    def _get_close_prices(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """获取收盘价序列"""
        if 'close' in data.columns:
            return data['close']
        elif 'S_DQ_CLOSE' in data.columns:
            return data['S_DQ_CLOSE']
        elif 'CLOSE' in data.columns:
            return data['CLOSE']
        return None

    def compute_with_details(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        计算信号并返回详细信息（用于调试和分析）

        Returns:
            (signals, details_df): 信号序列和详细信息DataFrame
        """
        prices = self._get_close_prices(data)
        if prices is None or len(prices) < self.slow_period + 1:
            return pd.Series(dtype=float), pd.DataFrame()

        # 计算均线
        fast_ma = self._calculate_ma(prices, self.fast_period)
        slow_ma = self._calculate_ma(prices, self.slow_period)

        # 计算交叉
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        # 逐日计算
        n = len(prices)
        signals = pd.Series(index=prices.index, dtype=float)
        signals[:] = 0

        ref_prices = pd.Series(index=prices.index, dtype=float)
        trade_types = pd.Series(index=prices.index, dtype=str)
        trade_types[:] = ''

        position = 0
        reference_price = None

        for i in range(self.slow_period, n):
            current_price = prices.iloc[i]

            if golden_cross.iloc[i]:
                if position == 1:
                    reference_price = current_price
                    trade_types.iloc[i] = 'update_long_ref'
                else:
                    position = 1
                    reference_price = current_price
                    trade_types.iloc[i] = 'open_long'

            elif death_cross.iloc[i]:
                if position == -1:
                    reference_price = current_price
                    trade_types.iloc[i] = 'update_short_ref'
                else:
                    position = -1
                    reference_price = current_price
                    trade_types.iloc[i] = 'open_short'

            # 平仓检查
            if position == 1 and reference_price is not None:
                if current_price < reference_price:
                    position = 0
                    reference_price = None
                    trade_types.iloc[i] = 'stop_long'

            elif position == -1 and reference_price is not None:
                if current_price > reference_price:
                    position = 0
                    reference_price = None
                    trade_types.iloc[i] = 'stop_short'

            signals.iloc[i] = position
            ref_prices.iloc[i] = reference_price

        # 构建详细信息
        details = pd.DataFrame({
            'price': prices,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'golden_cross': golden_cross,
            'death_cross': death_cross,
            'signal': signals,
            'ref_price': ref_prices,
            'trade_type': trade_types
        })

        return signals, details


# 便捷工厂函数
def create_wma_crossover_factor(
    fast_period: int = 10,
    slow_period: int = 20
) -> MACrossoverInnovationFactor:
    """创建WMA均线交叉因子（推荐配置）"""
    return MACrossoverInnovationFactor(
        fast_period=fast_period,
        slow_period=slow_period,
        ma_type='WMA'
    )


def create_sma_crossover_factor(
    fast_period: int = 10,
    slow_period: int = 20
) -> MACrossoverInnovationFactor:
    """创建SMA均线交叉因子"""
    return MACrossoverInnovationFactor(
        fast_period=fast_period,
        slow_period=slow_period,
        ma_type='SMA'
    )


def create_ema_crossover_factor(
    fast_period: int = 10,
    slow_period: int = 20
) -> MACrossoverInnovationFactor:
    """创建EMA均线交叉因子"""
    return MACrossoverInnovationFactor(
        fast_period=fast_period,
        slow_period=slow_period,
        ma_type='EMA'
    )


def create_llt_crossover_factor(
    fast_period: int = 10,
    slow_period: int = 20
) -> MACrossoverInnovationFactor:
    """创建LLT均线交叉因子"""
    return MACrossoverInnovationFactor(
        fast_period=fast_period,
        slow_period=slow_period,
        ma_type='LLT'
    )
