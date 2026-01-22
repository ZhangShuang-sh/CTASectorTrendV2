#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势跟踪配对交易因子

基于价格比率趋势的配对交易策略，使用快慢均线交叉判断趋势方向和强度。
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from core.factors.pair_trading.base import PairTradingFactorBase
from core.factors.utils.math_utils import (
    moving_average,
    calculate_correlation_matrix,
)


class TrendFollowingPairFactor(PairTradingFactorBase):
    """
    趋势跟踪配对截面因子

    对于每个品种，找到其最佳配对品种，计算价格比率的趋势强度，
    作为该品种的截面得分。

    得分含义:
    - 正值: 该品种相对配对品种处于上升趋势
    - 负值: 该品种相对配对品种处于下降趋势
    - 接近 0: 无明显趋势

    计算方法:
    使用快慢均线交叉判断趋势方向和强度。
    """

    def __init__(
        self,
        name: str = "TrendFollowingPairFactor",
        window: int = 60,
        fast_period: int = 10,
        slow_period: int = 30,
        min_correlation: float = 0.3,
    ):
        """
        Args:
            name: 因子名称
            window: 数据回看窗口
            fast_period: 快速均线周期
            slow_period: 慢速均线周期
            min_correlation: 最小相关性阈值
        """
        super().__init__(
            name=name,
            window=window,
            entry_threshold=0.0,
            exit_threshold=0.0
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_correlation = min_correlation

        self._params.update({
            'fast_period': fast_period,
            'slow_period': slow_period,
            'min_correlation': min_correlation
        })

    def set_params(self, **kwargs) -> 'TrendFollowingPairFactor':
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
        计算配对趋势信号

        Args:
            df1: 第一个资产的历史数据
            df2: 第二个资产的历史数据
            date: 当前日期

        Returns:
            float: 趋势强度得分 [-1, 1]
        """
        # 获取价格
        close_col = 'close' if 'close' in df1.columns else 'S_DQ_CLOSE'
        if close_col not in df1.columns or close_col not in df2.columns:
            return 0.0

        if date is not None:
            df1 = df1[df1.index <= date]
            df2 = df2[df2.index <= date]

        if len(df1) < self.slow_period + 5 or len(df2) < self.slow_period + 5:
            return 0.0

        prices1 = df1[close_col].values[-self.window:]
        prices2 = df2[close_col].values[-self.window:]

        return self._compute_trend_score(prices1, prices2)

    def calculate_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """
        计算全市场的趋势跟踪配对因子得分。

        Args:
            date: 当前截面日期
            universe_data: 全市场数据 {ticker: DataFrame}

        Returns:
            pd.Series: 每个品种的因子得分
        """
        scores = {}

        # 获取所有品种的价格数据
        prices_dict = self._extract_prices(date, universe_data)

        if len(prices_dict) < 2:
            return pd.Series(dtype=float)

        # 计算品种间相关性 (使用收益率)
        returns_dict = {
            ticker: np.diff(np.log(prices))
            for ticker, prices in prices_dict.items()
            if len(prices) > 1
        }
        correlations = calculate_correlation_matrix(returns_dict, min_periods=20)

        # 对每个品种计算趋势得分
        for ticker in prices_dict.keys():
            # 找到最佳配对
            best_pair = self._find_best_pair(ticker, correlations)

            if best_pair is None:
                scores[ticker] = 0.0
                continue

            # 计算趋势得分
            score = self._compute_trend_score(
                prices_dict[ticker],
                prices_dict[best_pair]
            )

            scores[ticker] = score

        return pd.Series(scores)

    def _extract_prices(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """提取每个品种的价格序列"""
        prices_dict = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < self.window:
                continue

            df_until = df[df.index <= date]
            if len(df_until) < self.window:
                continue

            close_col = 'close' if 'close' in df_until.columns else 'S_DQ_CLOSE'
            if close_col in df_until.columns:
                prices = df_until[close_col].values[-self.window:]
                if len(prices) >= self.slow_period + 10:
                    prices_dict[ticker] = prices

        return prices_dict

    def _find_best_pair(
        self,
        ticker: str,
        correlations: Dict[Tuple[str, str], float]
    ) -> Optional[str]:
        """找到最佳配对"""
        best_pair = None
        best_corr = self.min_correlation

        for (t1, t2), corr in correlations.items():
            if t1 == ticker and corr > best_corr:
                best_corr = corr
                best_pair = t2

        return best_pair

    def _compute_trend_score(
        self,
        prices1: np.ndarray,
        prices2: np.ndarray
    ) -> float:
        """
        计算趋势得分

        Returns:
            float: 趋势强度得分 [-1, 1]
        """
        min_len = min(len(prices1), len(prices2))
        p1 = prices1[-min_len:]
        p2 = prices2[-min_len:]

        if min_len < self.slow_period + 5:
            return 0.0

        # 计算价格比率
        ratio = p1 / np.maximum(p2, 1e-10)

        # 计算快慢均线 (使用共享工具)
        fast_ma = moving_average(ratio, self.fast_period)
        slow_ma = moving_average(ratio, self.slow_period)

        if len(fast_ma) == 0 or len(slow_ma) == 0:
            return 0.0

        # 取最新的均线差异
        ma_diff = (fast_ma[-1] - slow_ma[-1]) / slow_ma[-1]

        # 计算趋势持续性（均线斜率）
        if len(fast_ma) >= 5:
            trend_slope = (fast_ma[-1] - fast_ma[-5]) / fast_ma[-5]
        else:
            trend_slope = 0.0

        # 综合得分
        score = ma_diff * 10 + trend_slope * 5  # 放大信号

        return float(np.clip(score, -1.0, 1.0))
