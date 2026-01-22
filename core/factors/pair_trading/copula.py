#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Copula Pair Trading Factor

基于 Copula 模型的配对交易策略，利用条件概率偏离度生成交易信号。

参考: 报告《基于Copula的配对交易策略》

Source: V1 factors/pair_trading/copula.py (100% logic preserved)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from core.factors.pair_trading.base import UniversePairTradingFactorBase
from core.factors.utils.math_utils import (
    CopulaUtils,
    moving_average,
    calculate_correlation_matrix,
)
from core.factors.registry import register


@register('CopulaPairFactor')
class CopulaPairFactor(UniversePairTradingFactorBase):
    """
    Copula 配对交易截面因子

    对于每个品种，找到其最佳配对品种，使用 Copula 模型计算
    条件概率偏离度，作为该品种的截面得分。

    得分含义:
    - 正值: 该品种相对配对品种被低估，应做多
    - 负值: 该品种相对配对品种被高估，应做空
    - 接近 0: 无明显偏离

    参考: 报告《基于Copula的配对交易策略》
    """

    def __init__(
        self,
        name: str = "CopulaPairFactor",
        window: int = 60,
        correlation_window: int = 60,
        min_correlation: float = 0.5,
        entry_threshold: float = 0.2,
    ):
        """
        Args:
            name: 因子名称
            window: Copula 回看窗口
            correlation_window: 相关性计算窗口
            min_correlation: 最小相关性阈值（用于配对筛选）
            entry_threshold: 入场阈值
        """
        super().__init__(name=name, window=window, min_correlation=min_correlation)
        self.correlation_window = correlation_window
        self.entry_threshold = entry_threshold

        # 缓存配对信息
        self._pair_cache: Dict[str, str] = {}
        self._correlation_cache: Dict[Tuple[str, str], float] = {}

        self._params.update({
            'correlation_window': correlation_window,
            'entry_threshold': entry_threshold
        })

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """
        计算全市场的 Copula 配对因子得分。

        Args:
            universe_data: 全市场数据 {ticker: DataFrame}
            date: 当前截面日期 (可选，默认使用最新日期)
            **kwargs: 额外参数

        Returns:
            pd.Series: 每个品种的因子得分
        """
        if not self.validate_inputs(universe_data):
            return pd.Series(dtype=float)

        # 获取所有品种的收益率数据
        returns_dict = self._extract_returns(date, universe_data)

        if len(returns_dict) < 2:
            return pd.Series(dtype=float)

        scores = {}

        # 计算品种间相关性矩阵 (使用共享工具)
        correlations = calculate_correlation_matrix(returns_dict, min_periods=20)

        # 对每个品种计算 Copula 得分
        for ticker in returns_dict.keys():
            # 找到最佳配对
            best_pair = self._find_best_pair(ticker, correlations)

            if best_pair is None:
                scores[ticker] = 0.0
                continue

            # 计算 Copula 条件概率偏离
            score = self._compute_copula_score(
                returns_dict[ticker],
                returns_dict[best_pair]
            )

            scores[ticker] = score

        return pd.Series(scores)

    # V1 兼容方法
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """V1兼容方法 - 调用 calculate()"""
        return self.calculate(universe_data, date=date)

    def _extract_returns(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """提取每个品种的收益率序列"""
        returns_dict = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < self.window:
                continue

            # 筛选截止到 date 的数据
            if date is not None:
                # 支持多种索引格式
                if 'TRADE_DT' in df.columns:
                    df_until = df[df['TRADE_DT'] <= date]
                elif isinstance(df.index, pd.DatetimeIndex):
                    df_until = df[df.index <= date]
                else:
                    df_until = df
            else:
                df_until = df

            if len(df_until) < self.window:
                continue

            # 计算收益率 - 支持多种列名
            if 'returns' in df_until.columns:
                returns = df_until['returns'].dropna().values[-self.window:]
            elif 'S_DQ_CLOSE' in df_until.columns:
                returns = df_until['S_DQ_CLOSE'].pct_change().dropna().values[-self.window:]
            elif 'close' in df_until.columns:
                returns = df_until['close'].pct_change().dropna().values[-self.window:]
            else:
                continue

            if len(returns) >= self.window - 10:
                returns_dict[ticker] = returns

        return returns_dict

    def _find_best_pair(
        self,
        ticker: str,
        correlations: Dict[Tuple[str, str], float]
    ) -> Optional[str]:
        """找到指定品种的最佳配对"""
        best_pair = None
        best_corr = self.min_correlation

        for (t1, t2), corr in correlations.items():
            if t1 == ticker and corr > best_corr:
                best_corr = corr
                best_pair = t2

        return best_pair

    def _compute_copula_score(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray
    ) -> float:
        """
        计算 Copula 条件概率偏离得分

        Returns:
            float: 偏离得分 [-1, 1]
                   正值表示品种1相对被低估
                   负值表示品种1相对被高估
        """
        # 对齐长度
        min_len = min(len(returns1), len(returns2))
        r1 = returns1[-min_len:]
        r2 = returns2[-min_len:]

        if min_len < 20:
            return 0.0

        # 计算 ECDF (使用共享工具)
        ecdf1 = CopulaUtils.calculate_ecdf(r1[:-1])
        ecdf2 = CopulaUtils.calculate_ecdf(r2[:-1])

        # 转换为均匀分布
        u = ecdf1(r1[-1])
        v = ecdf2(r2[-1])

        # 限制在合理范围
        u = np.clip(u, 0.01, 0.99)
        v = np.clip(v, 0.01, 0.99)

        # 历史数据用于拟合 Copula
        historical_u = np.array([ecdf1(r) for r in r1[:-1]])
        historical_v = np.array([ecdf2(r) for r in r2[:-1]])
        historical_u = np.clip(historical_u, 0.01, 0.99)
        historical_v = np.clip(historical_v, 0.01, 0.99)

        # 拟合 Copula (使用共享工具)
        copula_type, theta = CopulaUtils.fit_copula(historical_u, historical_v)

        if copula_type is None or theta is None:
            return 0.0

        # 计算条件概率 (使用共享工具)
        try:
            cond_prob_func = CopulaUtils.get_conditional_prob_func(copula_type)
            cond_prob = cond_prob_func(u, v, theta)
        except Exception:
            return 0.0

        # 将条件概率转换为得分
        # cond_prob < 0.5: 品种1相对被低估 -> 正分
        # cond_prob > 0.5: 品种1相对被高估 -> 负分
        score = (0.5 - cond_prob) * 2  # 映射到 [-1, 1]

        # 应用阈值
        if abs(score) < self.entry_threshold:
            score = 0.0

        return float(np.clip(score, -1.0, 1.0))
