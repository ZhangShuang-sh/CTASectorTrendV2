#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
截面动量类因子实现。

包含:
- MomentumRank: 截面动量排名
- TermStructure: 期限结构/展期收益

参考: 报告10《CTA风格因子手册：动量类》
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from core.factors.cross_sectional.base import CrossSectionalFactorBase


class MomentumRank(CrossSectionalFactorBase):
    """
    截面动量排名因子 (Cross-Sectional Momentum Rank)。

    逻辑:
    - 计算过去 N 日的累计收益率
    - 可选: 剔除近期波动率过高的品种 (风险调整后动量)
    - 对所有品种进行排序，输出归一化分数 (0~1)
    """

    def __init__(
        self,
        name: str = "MomentumRank",
        window: int = 20,
        vol_filter_window: int = 20,
        vol_filter_threshold: Optional[float] = None
    ):
        """
        Args:
            name: 因子名称
            window: 动量回看窗口 (默认20日)
            vol_filter_window: 波动率计算窗口
            vol_filter_threshold: 波动率阈值，超过此值的品种将被剔除 (None表示不过滤)
        """
        super().__init__(name=name, window=window)
        self.vol_filter_window = vol_filter_window
        self.vol_filter_threshold = vol_filter_threshold

    def set_params(self, **kwargs) -> 'MomentumRank':
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def calculate(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """
        计算全市场的截面动量排名。

        Args:
            date: 当前回测截面的日期
            universe_data: 全市场数据字典
                Key: Ticker (如 'RB.SHF')
                Value: 该品种截至当前日期的历史数据 DataFrame

        Returns:
            pd.Series: 归一化后的动量排名分数 (0~1)
        """
        momentum_values = {}
        volatility_values = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < self.window:
                continue

            # 确保数据按日期排序
            df = df.sort_index()

            # 取截止到date的数据
            df_until_date = df[df.index <= date]
            if len(df_until_date) < self.window:
                continue

            # 计算累计收益率 (使用close列)
            close_col = 'close' if 'close' in df_until_date.columns else 'S_DQ_CLOSE'
            if close_col not in df_until_date.columns:
                continue

            close_prices = df_until_date[close_col].iloc[-self.window:]
            if close_prices.iloc[0] == 0:
                continue

            cumulative_return = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
            momentum_values[ticker] = cumulative_return

            # 计算波动率 (如果需要过滤)
            if self.vol_filter_threshold is not None:
                returns = df_until_date[close_col].pct_change().dropna()
                if len(returns) >= self.vol_filter_window:
                    vol = returns.iloc[-self.vol_filter_window:].std() * np.sqrt(252)
                    volatility_values[ticker] = vol

        if not momentum_values:
            return pd.Series(dtype=float)

        # 转为Series
        momentum_series = pd.Series(momentum_values)

        # 波动率过滤
        if self.vol_filter_threshold is not None and volatility_values:
            vol_series = pd.Series(volatility_values)
            valid_tickers = vol_series[vol_series <= self.vol_filter_threshold].index
            momentum_series = momentum_series.loc[
                momentum_series.index.intersection(valid_tickers)
            ]

        if len(momentum_series) == 0:
            return pd.Series(dtype=float)

        # Rank归一化到 [0, 1]
        ranks = momentum_series.rank(pct=True)

        return ranks


class TermStructure(CrossSectionalFactorBase):
    """
    期限结构因子 (Term Structure / Roll Yield)。

    逻辑:
    - 做多贴水(Backwardation)品种，做空升水(Contango)品种
    - 计算展期收益率: TS = (P_main - P_next) / P_next
    - 输出 Z-Score 标准化值
    """

    def __init__(
        self,
        name: str = "TermStructure",
        window: int = 1,
        zscore_window: int = 60
    ):
        """
        Args:
            name: 因子名称
            window: 计算展期收益的窗口 (默认1日，即当日值)
            zscore_window: Z-Score标准化的回看窗口
        """
        super().__init__(name=name, window=window)
        self.zscore_window = zscore_window

    def set_params(self, **kwargs) -> 'TermStructure':
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def calculate(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """
        计算全市场的期限结构因子。

        注意: 需要数据中包含 'close' (主力合约价格) 和 'close_next' (次主力合约价格) 列。

        Args:
            date: 当前回测截面的日期
            universe_data: 全市场数据字典

        Returns:
            pd.Series: Z-Score标准化后的期限结构因子值
        """
        term_structure_values = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < 1:
                continue

            # 确保数据按日期排序
            df = df.sort_index()

            # 取截止到date的数据
            df_until_date = df[df.index <= date]
            if len(df_until_date) < 1:
                continue

            # 获取最新数据行
            latest = df_until_date.iloc[-1]

            # 尝试获取主力和次主力价格
            p_main = None
            p_next = None

            if 'close' in latest.index:
                p_main = latest['close']
            elif 'S_DQ_CLOSE' in latest.index:
                p_main = latest['S_DQ_CLOSE']
            elif 'settle' in latest.index:
                p_main = latest['settle']

            if 'close_next' in latest.index:
                p_next = latest['close_next']
            elif 'settle_next' in latest.index:
                p_next = latest['settle_next']
            elif 'next_close' in latest.index:
                p_next = latest['next_close']

            # 如果没有次主力价格，跳过该品种
            if p_main is None or p_next is None or p_next == 0:
                continue

            # 计算展期收益率
            # 正值表示贴水 (Backwardation)，负值表示升水 (Contango)
            term_structure = (p_main - p_next) / p_next
            term_structure_values[ticker] = term_structure

        if not term_structure_values:
            return pd.Series(dtype=float)

        # 转为Series
        ts_series = pd.Series(term_structure_values)

        # Z-Score标准化
        mean_val = ts_series.mean()
        std_val = ts_series.std()

        if std_val == 0 or np.isnan(std_val):
            # 如果标准差为0，返回0
            return pd.Series(0.0, index=ts_series.index)

        zscore = (ts_series - mean_val) / std_val

        return zscore
