#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - VCRR Factor (VWAP-Close Relative Rank)

VCRR因子（VWAP-Close Relative Rank）

来源: 中泰期货研究所量化CTA因子库

理论背景:
- VWAP (Volume Weighted Average Price) 是成交量加权平均价格
- VCRR比较VWAP与收盘价的关系
- 当VWAP > Close时，说明大资金买入价高于当前价（潜在支撑）
- 当VWAP < Close时，说明大资金买入价低于当前价（潜在压力）

计算逻辑:
1. VWAP = amount / volume / unit * 10000 (或简化为 amount / volume)
2. VCRR = (VWAP - Close).rank(pct=True) / (VWAP + Close).rank(pct=True)
3. 该因子是截面因子，对所有品种进行排名比较

Source: V1 factors/cross_sectional/vcrr.py (100% logic preserved)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from core.factors.cross_sectional.base import CrossSectionalFactor
from core.factors.time_series.base import TimeSeriesFactor
from core.factors.registry import register


@register('VCRRFactor')
class VCRRFactor(CrossSectionalFactor):
    """
    VCRR因子（VWAP-Close相对排名因子）

    计算逻辑:
    1. 对每个品种计算 (VWAP - Close) 值
    2. 对所有品种进行截面百分位排名
    3. 同样计算 (VWAP + Close) 的截面排名
    4. VCRR = diff_rank / sum_rank

    信号解读:
    - 高VCRR值: VWAP相对于Close较高，可能有支撑
    - 低VCRR值: VWAP相对于Close较低，可能有压力

    参数:
    - use_amount: 是否使用成交额计算VWAP（默认True）
    - multiplier: 合约乘数（简化版本不使用）
    """

    def __init__(
        self,
        use_amount: bool = True,
        name: str = None
    ):
        """
        Args:
            use_amount: 是否使用成交额计算VWAP
        """
        if name is None:
            name = "VCRR"

        super().__init__(name=name, window=20)  # 只需要当天数据，但保留一定窗口
        self.use_amount = use_amount

        self._params.update({
            'use_amount': use_amount
        })

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """
        计算单个品种的VWAP

        Args:
            df: 品种数据，必须包含volume列，可选amount列

        Returns:
            float: VWAP值
        """
        if len(df) == 0:
            return np.nan

        # 获取最后一行数据
        last_row = df.iloc[-1]

        # Try different column names for close
        close_col = 'close' if 'close' in df.columns else 'S_DQ_CLOSE'
        vol_col = 'volume' if 'volume' in df.columns else 'S_DQ_VOLUME'

        close = last_row.get(close_col, np.nan)
        volume = last_row.get(vol_col, np.nan)

        if pd.isna(close) or pd.isna(volume) or volume == 0:
            return np.nan

        if self.use_amount and 'amount' in df.columns:
            amount = last_row.get('amount', np.nan)
            if pd.isna(amount):
                return close  # 回退到收盘价
            vwap = amount / volume
        else:
            # 简化版本：用OHLC的平均作为近似
            high_col = 'high' if 'high' in df.columns else 'S_DQ_HIGH'
            low_col = 'low' if 'low' in df.columns else 'S_DQ_LOW'
            open_col = 'open' if 'open' in df.columns else 'S_DQ_OPEN'

            high = last_row.get(high_col, close)
            low = last_row.get(low_col, close)
            open_price = last_row.get(open_col, close)
            vwap = (high + low + close + open_price) / 4

        return vwap

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """
        计算全市场的VCRR因子值

        Args:
            universe_data: 全市场数据 Dict[ticker, DataFrame]
            date: 当前日期

        Returns:
            pd.Series: 各品种的VCRR因子值，索引为ticker
        """
        vwap_dict = {}
        close_dict = {}

        for ticker, df in universe_data.items():
            if len(df) == 0:
                continue

            # 确保数据包含当前日期
            if date is not None and date in df.index:
                data_up_to_date = df.loc[:date]
            else:
                data_up_to_date = df

            if len(data_up_to_date) == 0:
                continue

            # 计算VWAP和Close
            vwap = self._calculate_vwap(data_up_to_date)

            close_col = 'close' if 'close' in data_up_to_date.columns else 'S_DQ_CLOSE'
            close = data_up_to_date[close_col].iloc[-1] if close_col in data_up_to_date.columns else np.nan

            if not pd.isna(vwap) and not pd.isna(close):
                vwap_dict[ticker] = vwap
                close_dict[ticker] = close

        if len(vwap_dict) == 0:
            return pd.Series(dtype=float)

        # 转换为Series
        vwap_series = pd.Series(vwap_dict)
        close_series = pd.Series(close_dict)

        # 计算差值和总和
        diff = vwap_series - close_series
        sum_val = vwap_series + close_series

        # 截面百分位排名
        diff_rank = diff.rank(pct=True)
        sum_rank = sum_val.rank(pct=True)

        # 计算VCRR
        vcrr = diff_rank / (sum_rank + 1e-10)

        # 标准化到[-1, 1]范围
        # 原始VCRR范围约为[0, inf)，需要归一化
        vcrr_normalized = (vcrr - vcrr.mean()) / (vcrr.std() + 1e-10)
        vcrr_normalized = vcrr_normalized.clip(-3, 3) / 3  # 映射到[-1, 1]

        return vcrr_normalized

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


@register('VCRRTimeSeriesFactor')
class VCRRTimeSeriesFactor(TimeSeriesFactor):
    """
    VCRR时序版本因子

    针对单个品种，计算VWAP与Close的相对关系，
    并与历史值比较进行标准化。

    信号解读:
    - 正值: 当前VWAP相对于Close较高（相对历史）
    - 负值: 当前VWAP相对于Close较低（相对历史）
    """

    def __init__(
        self,
        lookback_period: int = 20,
        use_amount: bool = True,
        name: str = None
    ):
        """
        Args:
            lookback_period: 历史比较窗口
            use_amount: 是否使用成交额计算VWAP
        """
        window = max(lookback_period + 10, 40)

        if name is None:
            name = f"VCRR_TS_{lookback_period}"

        super().__init__(name=name, window=window)

        self.lookback_period = lookback_period
        self.use_amount = use_amount

        self._params.update({
            'lookback_period': lookback_period,
            'use_amount': use_amount
        })

    def _calculate_vwap_series(self, data: pd.DataFrame) -> pd.Series:
        """计算VWAP序列"""
        vol_col = 'volume' if 'volume' in data.columns else 'S_DQ_VOLUME'
        high_col = 'high' if 'high' in data.columns else 'S_DQ_HIGH'
        low_col = 'low' if 'low' in data.columns else 'S_DQ_LOW'
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        open_col = 'open' if 'open' in data.columns else 'S_DQ_OPEN'

        if self.use_amount and 'amount' in data.columns:
            vwap = data['amount'] / (data[vol_col] + 1e-10)
        else:
            # 使用OHLC平均近似
            vwap = (data[high_col] + data[low_col] + data[close_col] + data[open_col]) / 4
        return vwap

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算VCRR因子值

        Args:
            data: 必须包含 'close', 'volume' 列

        Returns:
            float: 因子值
        """
        if len(data) < self.window:
            return 0.0

        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'

        vwap = self._calculate_vwap_series(data)
        close = data[close_col]

        # 计算VWAP-Close差值
        diff = vwap - close

        # 与历史比较（滚动z-score）
        diff_mean = diff.rolling(self.lookback_period).mean()
        diff_std = diff.rolling(self.lookback_period).std()

        z_score = (diff - diff_mean) / (diff_std + 1e-10)

        result = z_score.iloc[-1]
        if pd.isna(result) or np.isinf(result):
            return 0.0

        return float(np.clip(result, -3.0, 3.0) / 3.0)  # 归一化到[-1, 1]

    def compute(self, data: pd.DataFrame) -> float:
        """Backward compatible method name"""
        return self.calculate(data)

    def calculate_series(self, data: pd.DataFrame) -> pd.Series:
        """计算完整因子序列"""
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'

        vwap = self._calculate_vwap_series(data)
        close = data[close_col]

        diff = vwap - close

        diff_mean = diff.rolling(self.lookback_period).mean()
        diff_std = diff.rolling(self.lookback_period).std()

        z_score = (diff - diff_mean) / (diff_std + 1e-10)
        z_score = z_score.replace([np.inf, -np.inf], np.nan)
        z_score = z_score.clip(-3.0, 3.0) / 3.0

        return z_score

    def compute_series(self, data: pd.DataFrame) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate_series(data)


# 便捷工厂函数
def create_vcrr_factor(**kwargs) -> VCRRFactor:
    """创建VCRR截面因子"""
    return VCRRFactor(**kwargs)


def create_vcrr_ts_factor(
    lookback_period: int = 20,
    **kwargs
) -> VCRRTimeSeriesFactor:
    """创建VCRR时序因子"""
    return VCRRTimeSeriesFactor(lookback_period=lookback_period, **kwargs)
