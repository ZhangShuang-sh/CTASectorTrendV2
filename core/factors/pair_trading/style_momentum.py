#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风格动量配对交易因子

基于华泰期货《CTA研究系列之二十二：风格动量下的股指期货跨品种套利策略》

核心逻辑：
- 风格动量效应：历史表现较强的资产，短期内会延续强势
- 比较配对资产的日收益率，做多强势资产，做空弱势资产
- 纯动量跟随策略，无需参数优化

信号生成：
- 若资产A收益率 > 资产B收益率 → 做多A，做空B
- 若资产A收益率 < 资产B收益率 → 做空A，做多B

原报告业绩（股指期货，无杠杆）：
- 50-500套利：年化32.41%，夏普2.75，最大回撤-9.07%
- 50-300套利：年化6.69%，夏普1.72，最大回撤-7.80%
- 300-500套利：年化20.15%，夏普2.59，最大回撤-5.00%
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from core.factors.pair_trading.base import PairTradingFactorBase


class StyleMomentumPairFactor(PairTradingFactorBase):
    """
    风格动量配对交易因子

    核心逻辑：
    - 比较配对资产的日收益率
    - 做多当日相对强势的资产，做空相对弱势的资产
    - 纯动量跟随，追涨杀跌

    信号含义：
    - +1: 资产1收益 > 资产2收益 → 做多资产1，做空资产2
    - -1: 资产1收益 < 资产2收益 → 做空资产1，做多资产2
    - 0: 收益相等或数据不足 → 无信号
    """

    def __init__(
        self,
        name: str = "StyleMomentumPairFactor",
        lookback: int = 1,
        min_history: int = 2,
        signal_strength: float = 1.0,
        use_return_magnitude: bool = False,
        window: int = 20
    ):
        """
        Args:
            name: 因子名称
            lookback: 收益率回看期（天），默认1表示日频
            min_history: 最小历史数据要求
            signal_strength: 信号强度缩放因子
            use_return_magnitude: 是否使用收益率差异幅度调整信号强度
            window: 回看窗口（用于计算历史波动率等）
        """
        super().__init__(
            name=name,
            window=window,
            entry_threshold=0.0,  # 风格动量无阈值
            exit_threshold=0.0
        )
        self.lookback = lookback
        self.min_history = min_history
        self.signal_strength = signal_strength
        self.use_return_magnitude = use_return_magnitude

        self._params.update({
            'lookback': lookback,
            'min_history': min_history,
            'signal_strength': signal_strength,
            'use_return_magnitude': use_return_magnitude
        })

    def set_params(self, **kwargs) -> 'StyleMomentumPairFactor':
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def calculate(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        date: pd.Timestamp = None
    ) -> float:
        """
        计算风格动量配对信号

        Args:
            df1: 第一个资产的历史数据（需包含 'close' 列）
            df2: 第二个资产的历史数据（需包含 'close' 列）
            date: 当前日期

        Returns:
            float: 信号值
                   +1.0: 做多资产1，做空资产2
                   -1.0: 做空资产1，做多资产2
                   0.0: 无信号
        """
        # 数据预处理
        if date is not None and isinstance(df1.index, pd.DatetimeIndex):
            df1_until = df1[df1.index <= date]
            df2_until = df2[df2.index <= date]
        else:
            df1_until = df1
            df2_until = df2

        # 检查数据充足性
        if len(df1_until) < self.min_history or len(df2_until) < self.min_history:
            return 0.0

        # 获取收盘价序列
        close1 = self._get_close_prices(df1_until)
        close2 = self._get_close_prices(df2_until)

        if close1 is None or close2 is None:
            return 0.0

        if len(close1) < self.lookback + 1 or len(close2) < self.lookback + 1:
            return 0.0

        # 计算收益率
        ret1 = self._calculate_return(close1, self.lookback)
        ret2 = self._calculate_return(close2, self.lookback)

        if pd.isna(ret1) or pd.isna(ret2):
            return 0.0

        # 生成信号
        signal = self._generate_signal(ret1, ret2)

        return signal

    def _get_close_prices(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """获取收盘价序列"""
        if 'close' in df.columns:
            return df['close']
        elif 'S_DQ_CLOSE' in df.columns:
            return df['S_DQ_CLOSE']
        elif 'CLOSE' in df.columns:
            return df['CLOSE']
        return None

    def _calculate_return(self, prices: pd.Series, lookback: int) -> float:
        """计算收益率"""
        if len(prices) < lookback + 1:
            return np.nan

        current_price = prices.iloc[-1]
        past_price = prices.iloc[-lookback - 1]

        if past_price == 0 or pd.isna(past_price) or pd.isna(current_price):
            return np.nan

        return (current_price - past_price) / past_price

    def _generate_signal(self, ret1: float, ret2: float) -> float:
        """
        根据收益率比较生成信号

        核心规则：
        - ret1 > ret2: 资产1更强 → 做多资产1，做空资产2 → +1
        - ret1 < ret2: 资产2更强 → 做空资产1，做多资产2 → -1
        - ret1 == ret2: 无明显差异 → 0
        """
        diff = ret1 - ret2

        if diff > 0:
            signal = 1.0
        elif diff < 0:
            signal = -1.0
        else:
            signal = 0.0

        # 可选：根据收益率差异幅度调整信号强度
        if self.use_return_magnitude and signal != 0:
            # 使用收益率差异的绝对值作为信号强度调整
            # 限制在合理范围内
            magnitude = min(abs(diff) * 100, 1.0)  # 1%差异对应满仓
            signal = signal * magnitude

        # 应用信号强度缩放
        signal = signal * self.signal_strength

        return float(np.clip(signal, -1.0, 1.0))

    def compute_series(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        计算完整的信号序列（用于回测）

        Args:
            df1: 第一个资产的历史数据
            df2: 第二个资产的历史数据
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.Series: 信号序列，索引为日期
        """
        # 获取收盘价
        close1 = self._get_close_prices(df1)
        close2 = self._get_close_prices(df2)

        if close1 is None or close2 is None:
            return pd.Series(dtype=float)

        # 对齐日期索引
        common_index = close1.index.intersection(close2.index)
        if len(common_index) < self.min_history:
            return pd.Series(dtype=float)

        close1 = close1.loc[common_index]
        close2 = close2.loc[common_index]

        # 计算收益率
        ret1 = close1.pct_change(self.lookback)
        ret2 = close2.pct_change(self.lookback)

        # 生成信号序列
        signals = pd.Series(index=common_index, dtype=float)

        for i in range(self.lookback, len(common_index)):
            r1 = ret1.iloc[i]
            r2 = ret2.iloc[i]

            if pd.isna(r1) or pd.isna(r2):
                signals.iloc[i] = 0.0
            else:
                signals.iloc[i] = self._generate_signal(r1, r2)

        # 应用日期范围过滤
        if start_date is not None:
            signals = signals[signals.index >= start_date]
        if end_date is not None:
            signals = signals[signals.index <= end_date]

        return signals


class EnhancedStyleMomentumPairFactor(StyleMomentumPairFactor):
    """
    增强版风格动量配对因子

    在基础版本上增加：
    1. 动量强度过滤：仅在动量差异显著时交易
    2. 波动率调整：根据波动率调整仓位
    3. 趋势确认：使用短期均线确认趋势
    """

    def __init__(
        self,
        name: str = "EnhancedStyleMomentumPairFactor",
        lookback: int = 1,
        min_diff_threshold: float = 0.001,
        vol_adjust: bool = True,
        vol_window: int = 20,
        trend_confirm: bool = False,
        trend_ma_window: int = 5,
        **kwargs
    ):
        """
        Args:
            name: 因子名称
            lookback: 收益率回看期
            min_diff_threshold: 最小收益率差异阈值（过滤噪音）
            vol_adjust: 是否进行波动率调整
            vol_window: 波动率计算窗口
            trend_confirm: 是否需要趋势确认
            trend_ma_window: 趋势确认均线窗口
        """
        super().__init__(name=name, lookback=lookback, **kwargs)
        self.min_diff_threshold = min_diff_threshold
        self.vol_adjust = vol_adjust
        self.vol_window = vol_window
        self.trend_confirm = trend_confirm
        self.trend_ma_window = trend_ma_window

        self._params.update({
            'min_diff_threshold': min_diff_threshold,
            'vol_adjust': vol_adjust,
            'vol_window': vol_window,
            'trend_confirm': trend_confirm,
            'trend_ma_window': trend_ma_window
        })

    def calculate(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        date: pd.Timestamp = None
    ) -> float:
        """增强版信号计算"""
        # 数据预处理
        if date is not None and isinstance(df1.index, pd.DatetimeIndex):
            df1_until = df1[df1.index <= date]
            df2_until = df2[df2.index <= date]
        else:
            df1_until = df1
            df2_until = df2

        min_required = max(self.min_history, self.vol_window, self.trend_ma_window)
        if len(df1_until) < min_required or len(df2_until) < min_required:
            return 0.0

        close1 = self._get_close_prices(df1_until)
        close2 = self._get_close_prices(df2_until)

        if close1 is None or close2 is None:
            return 0.0

        # 计算收益率
        ret1 = self._calculate_return(close1, self.lookback)
        ret2 = self._calculate_return(close2, self.lookback)

        if pd.isna(ret1) or pd.isna(ret2):
            return 0.0

        # 收益率差异过滤
        diff = ret1 - ret2
        if abs(diff) < self.min_diff_threshold:
            return 0.0

        # 基础信号
        if diff > 0:
            signal = 1.0
        else:
            signal = -1.0

        # 趋势确认（可选）
        if self.trend_confirm:
            if not self._confirm_trend(close1, close2, signal):
                return 0.0

        # 波动率调整（可选）
        if self.vol_adjust:
            signal = self._adjust_by_volatility(close1, close2, signal)

        return float(np.clip(signal * self.signal_strength, -1.0, 1.0))

    def _confirm_trend(
        self,
        close1: pd.Series,
        close2: pd.Series,
        signal: float
    ) -> bool:
        """
        使用短期均线确认趋势

        确认逻辑：
        - 如果做多资产1（signal > 0），确认资产1短期均线向上
        - 如果做多资产2（signal < 0），确认资产2短期均线向上
        """
        if len(close1) < self.trend_ma_window or len(close2) < self.trend_ma_window:
            return True  # 数据不足时不过滤

        ma1 = close1.rolling(self.trend_ma_window).mean()
        ma2 = close2.rolling(self.trend_ma_window).mean()

        if signal > 0:
            # 做多资产1，确认资产1价格在均线上方
            return close1.iloc[-1] > ma1.iloc[-1]
        else:
            # 做多资产2，确认资产2价格在均线上方
            return close2.iloc[-1] > ma2.iloc[-1]

    def _adjust_by_volatility(
        self,
        close1: pd.Series,
        close2: pd.Series,
        signal: float
    ) -> float:
        """
        根据价差波动率调整信号强度

        波动率越高，信号强度越低（风险控制）
        """
        if len(close1) < self.vol_window or len(close2) < self.vol_window:
            return signal

        # 计算价差
        spread = close1 / close2

        # 计算价差波动率
        spread_ret = spread.pct_change()
        vol = spread_ret.rolling(self.vol_window).std().iloc[-1]

        if pd.isna(vol) or vol == 0:
            return signal

        # 使用目标波动率15%进行归一化
        target_vol = 0.15 / np.sqrt(252)  # 日度目标波动率
        vol_factor = target_vol / vol

        # 限制波动率调整范围
        vol_factor = np.clip(vol_factor, 0.5, 2.0)

        return signal * vol_factor
