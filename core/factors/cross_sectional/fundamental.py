#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Fundamental Cross-Sectional Factors

截面基本面类因子实现。

包含:
- MemberHoldings: 会员持仓净多空因子

Source: V1 factors/cross_sectional/fundamental.py (100% logic preserved)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable

from core.factors.cross_sectional.base import CrossSectionalFactor
from core.factors.registry import register


@register('MemberHoldings')
class MemberHoldings(CrossSectionalFactor):
    """
    会员持仓净多空因子 (Member Holdings Difference)。

    逻辑:
    - 基于交易所公布的前20名主力会员持仓数据
    - 计算净持仓占比: (Top20_Long - Top20_Short) / (Top20_Long + Top20_Short)
    - 可选: 引入"会员历史预测胜率"作为权重
    - 输出 Z-Score 或 Rank 标准化值

    参考: 报告07/10《会员持仓因子的预测正确性动量》
    """

    def __init__(
        self,
        name: str = "MemberHoldings",
        window: int = 1,
        use_win_rate_weight: bool = False,
        alt_data_loader: Optional[Callable] = None
    ):
        """
        Args:
            name: 因子名称
            window: 回看窗口
            use_win_rate_weight: 是否使用会员历史预测胜率加权
            alt_data_loader: 替代数据加载器，用于加载会员持仓数据
                             函数签名: (date: pd.Timestamp) -> pd.DataFrame
                             返回的DataFrame应包含列:
                             - 'ticker': 品种代码
                             - 'long_position': 多头持仓量
                             - 'short_position': 空头持仓量
                             - 'win_rate' (可选): 会员历史预测胜率
        """
        super().__init__(name=name, window=window)
        self.use_win_rate_weight = use_win_rate_weight
        self.alt_data_loader = alt_data_loader

    def set_params(self, **kwargs) -> 'MemberHoldings':
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """
        计算全市场的会员持仓因子。

        数据来源优先级:
        1. 如果提供了 alt_data_loader，使用外部数据
        2. 否则，尝试从 universe_data 的 DataFrame 中提取相关列

        Args:
            universe_data: 全市场数据字典
            date: 当前回测截面的日期

        Returns:
            pd.Series: Z-Score标准化后的会员持仓因子值
        """
        holdings_values = {}
        win_rates = {}

        # 方式1: 使用外部数据加载器
        if self.alt_data_loader is not None and date is not None:
            holdings_data = self.alt_data_loader(date)
            if holdings_data is not None and len(holdings_data) > 0:
                for _, row in holdings_data.iterrows():
                    ticker = row.get('ticker')
                    if ticker is None or ticker not in universe_data:
                        continue

                    long_pos = row.get('long_position', 0)
                    short_pos = row.get('short_position', 0)
                    total = long_pos + short_pos

                    if total == 0:
                        continue

                    net_ratio = (long_pos - short_pos) / total
                    holdings_values[ticker] = net_ratio

                    if self.use_win_rate_weight and 'win_rate' in row.index:
                        win_rates[ticker] = row['win_rate']

        # 方式2: 从 universe_data 提取
        if not holdings_values:
            for ticker, df in universe_data.items():
                if df is None or len(df) < 1:
                    continue

                df = df.sort_index()
                if date is not None:
                    df_until_date = df[df.index <= date]
                else:
                    df_until_date = df

                if len(df_until_date) < 1:
                    continue

                latest = df_until_date.iloc[-1]

                # 尝试不同的列名
                long_pos = None
                short_pos = None

                # 常见的列名变体
                long_columns = ['long_position', 'top20_long', 'long_oi', 'member_long']
                short_columns = ['short_position', 'top20_short', 'short_oi', 'member_short']

                for col in long_columns:
                    if col in latest.index and not pd.isna(latest[col]):
                        long_pos = latest[col]
                        break

                for col in short_columns:
                    if col in latest.index and not pd.isna(latest[col]):
                        short_pos = latest[col]
                        break

                if long_pos is None or short_pos is None:
                    continue

                total = long_pos + short_pos
                if total == 0:
                    continue

                net_ratio = (long_pos - short_pos) / total
                holdings_values[ticker] = net_ratio

        if not holdings_values:
            return pd.Series(dtype=float)

        # 转为Series
        holdings_series = pd.Series(holdings_values)

        # 应用胜率权重 (如果启用且有数据)
        if self.use_win_rate_weight and win_rates:
            win_rate_series = pd.Series(win_rates)
            common_idx = holdings_series.index.intersection(win_rate_series.index)
            if len(common_idx) > 0:
                holdings_series = holdings_series.loc[common_idx] * win_rate_series.loc[common_idx]

        # Z-Score标准化
        mean_val = holdings_series.mean()
        std_val = holdings_series.std()

        if std_val == 0 or np.isnan(std_val):
            return pd.Series(0.0, index=holdings_series.index)

        zscore = (holdings_series - mean_val) / std_val

        return zscore

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)
