#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级波动率类时序因子

来源: 中泰期货研究所量化CTA因子库

包含因子:
- DUVOLFactor: 下行波动率因子（Downside Volatility）
- RunsTestFactor: 基于游程检验的随机性因子

理论背景:
- DUVOL: 衡量下行风险与上行风险的不对称性
- Runs Test: 检测价格序列的随机性，低随机性暗示趋势存在
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import norm

from core.factors.time_series.base import TimeSeriesFactorBase


class DUVOLFactor(TimeSeriesFactorBase):
    """
    下行波动率因子（Downside Volatility）

    计算逻辑:
    1. 计算收益率序列（可选：减去滚动均值进行去趋势处理）
    2. 分离正收益和负收益
    3. 计算上行方差和下行方差
    4. Factor = log(下行方差/下行计数 / 上行方差/上行计数)

    信号解读:
    - 高值: 下行波动率相对较大（下跌风险高）-> 偏空
    - 低值: 上行波动率相对较大（上涨动能强）-> 偏多
    """

    def __init__(
        self,
        period: int = 150,
        demean: bool = False,
        name: str = None
    ):
        """
        Args:
            period: 波动率计算窗口
            demean: 是否对收益率去均值
        """
        window = max(period + 20, 200)

        if name is None:
            name = f"DUVOL_{period}"

        super().__init__(name=name, window=window)

        self.period = period
        self.demean = demean

        self._params.update({
            'period': period,
            'demean': demean
        })

    def calculate(self, data: pd.DataFrame) -> float:
        """
        计算DUVOL因子值

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            float: 因子值（正值偏空，负值偏多）
        """
        if len(data) < self.window:
            return 0.0

        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return 0.0

        prices = data[close_col]
        returns = prices.pct_change()

        if self.demean:
            # 减去滚动均值
            returns = returns - returns.rolling(self.period).mean()

        # 分离正负收益
        pos_rtn = returns.where(returns > 0, 0)
        neg_rtn = returns.where(returns < 0, 0)

        # 计算方差（平方和）
        pos_var = (pos_rtn ** 2).rolling(self.period).sum()
        neg_var = (neg_rtn ** 2).rolling(self.period).sum()

        # 计算计数
        pos_count = (returns > 0).rolling(self.period).sum()
        neg_count = (returns < 0).rolling(self.period).sum()

        # 计算标准化方差
        pos_std_var = pos_var / (pos_count - 1 + 1e-10)
        neg_std_var = neg_var / (neg_count - 1 + 1e-10)

        # 计算因子值（对数比）
        factor_value = np.log(neg_std_var / (pos_std_var + 1e-10) + 1e-10)

        result = factor_value.iloc[-1]
        if pd.isna(result) or np.isinf(result):
            return 0.0

        return float(np.clip(result, -5.0, 5.0))

    def compute_series(self, data: pd.DataFrame) -> pd.Series:
        """
        计算完整的因子序列

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            pd.Series: 因子值序列
        """
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return pd.Series(dtype=float)

        prices = data[close_col]
        returns = prices.pct_change()

        if self.demean:
            returns = returns - returns.rolling(self.period).mean()

        pos_rtn = returns.where(returns > 0, 0)
        neg_rtn = returns.where(returns < 0, 0)

        pos_var = (pos_rtn ** 2).rolling(self.period).sum()
        neg_var = (neg_rtn ** 2).rolling(self.period).sum()

        pos_count = (returns > 0).rolling(self.period).sum()
        neg_count = (returns < 0).rolling(self.period).sum()

        pos_std_var = pos_var / (pos_count - 1 + 1e-10)
        neg_std_var = neg_var / (neg_count - 1 + 1e-10)

        factor = np.log(neg_std_var / (pos_std_var + 1e-10) + 1e-10)

        factor = factor.replace([np.inf, -np.inf], np.nan)
        factor = factor.clip(-5.0, 5.0)

        return factor


class RunsTestFactor(TimeSeriesFactorBase):
    """
    游程检验因子（Runs Test Factor）

    计算逻辑:
    1. 将收益率转化为二元序列（涨=1，跌=0）
    2. 使用游程检验计算随机性p值
    3. p值低 -> 序列非随机 -> 存在趋势
    4. 结合趋势方向生成信号

    信号规则:
    - p_value < threshold AND MA上升 -> 做多
    - p_value < threshold AND MA下降 -> 做空
    """

    def __init__(
        self,
        runs_window: int = 10,
        ma_window: int = 60,
        p_threshold: float = 0.05,
        name: str = None
    ):
        """
        Args:
            runs_window: 游程检验窗口
            ma_window: MA趋势判断窗口
            p_threshold: p值阈值
        """
        window = max(ma_window + runs_window + 20, 100)

        if name is None:
            name = f"RunsTest_{runs_window}_{ma_window}"

        super().__init__(name=name, window=window)

        self.runs_window = runs_window
        self.ma_window = ma_window
        self.p_threshold = p_threshold

        self._params.update({
            'runs_window': runs_window,
            'ma_window': ma_window,
            'p_threshold': p_threshold
        })

    @staticmethod
    def _runs_test(sequence: np.ndarray) -> float:
        """
        游程检验

        Args:
            sequence: 二元序列（0和1）

        Returns:
            float: p值
        """
        if len(sequence) < 2:
            return 1.0

        sequence = np.array(sequence)
        n = len(sequence)
        n1 = np.sum(sequence)  # 1的个数
        n2 = n - n1  # 0的个数

        if n1 == 0 or n2 == 0:
            return 1.0  # 全为0或全为1，返回高p值

        # 计算游程数
        runs = np.sum(np.diff(sequence) != 0) + 1

        # 期望游程数
        expected_runs = (2 * n1 * n2) / n + 1

        # 游程标准差
        numerator = 2 * n1 * n2 * (2 * n1 * n2 - n)
        denominator = n ** 2 * (n - 1)

        if denominator <= 0 or numerator <= 0:
            return 1.0

        std_runs = np.sqrt(numerator / denominator)

        if std_runs == 0:
            return 1.0

        # Z分数
        z_score = (runs - expected_runs) / std_runs

        # 双尾p值
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        return p_value

    def calculate(self, data: pd.DataFrame) -> float:
        """
        计算游程检验因子值

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            float: 信号值 (+1=做多, -1=做空, 0=无信号)
        """
        if len(data) < self.window:
            return 0.0

        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return 0.0

        prices = data[close_col]
        returns = prices.pct_change()

        # 二元化收益率
        binary_returns = (returns >= 0).astype(int)

        # 计算p值（使用最近runs_window个数据）
        recent_binary = binary_returns.dropna().iloc[-self.runs_window:]
        if len(recent_binary) < self.runs_window:
            return 0.0

        p_value = self._runs_test(recent_binary.values)

        # 计算MA趋势
        ma = prices.rolling(self.ma_window).mean()
        ma_trend = ma.iloc[-1] > ma.iloc[-2] if len(ma) >= 2 else False

        # 生成信号
        if p_value < self.p_threshold:
            if ma_trend:
                return 1.0  # 做多
            else:
                return -1.0  # 做空
        else:
            return 0.0  # 无信号（序列随机）

    def compute_series(self, data: pd.DataFrame) -> pd.Series:
        """
        计算完整的因子序列

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            pd.Series: 信号序列
        """
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return pd.Series(dtype=float)

        prices = data[close_col]
        returns = prices.pct_change()
        binary_returns = (returns >= 0).astype(int)

        # 计算滚动p值
        def rolling_runs_test(x):
            return self._runs_test(x.values)

        p_values = binary_returns.rolling(self.runs_window).apply(
            rolling_runs_test, raw=False
        )

        # 计算MA趋势
        ma = prices.rolling(self.ma_window).mean()
        ma_trend = ma > ma.shift(1)

        # 生成信号
        signals = pd.Series(index=data.index, dtype=float)
        signals[:] = 0.0

        # p值低且MA上升 -> 做多
        long_condition = (p_values < self.p_threshold) & ma_trend
        # p值低且MA下降 -> 做空
        short_condition = (p_values < self.p_threshold) & ~ma_trend

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals

    def get_p_values(self, data: pd.DataFrame) -> pd.Series:
        """
        获取游程检验p值序列（用于分析）

        Returns:
            pd.Series: p值序列
        """
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return pd.Series(dtype=float)

        prices = data[close_col]
        returns = prices.pct_change()
        binary_returns = (returns >= 0).astype(int)

        def rolling_runs_test(x):
            return self._runs_test(x.values)

        p_values = binary_returns.rolling(self.runs_window).apply(
            rolling_runs_test, raw=False
        )

        return p_values


# 便捷工厂函数
def create_duvol_factor(
    period: int = 150,
    **kwargs
) -> DUVOLFactor:
    """创建DUVOL因子"""
    return DUVOLFactor(period=period, **kwargs)


def create_runs_test_factor(
    runs_window: int = 10,
    ma_window: int = 60,
    p_threshold: float = 0.05,
    **kwargs
) -> RunsTestFactor:
    """创建游程检验因子"""
    return RunsTestFactor(
        runs_window=runs_window,
        ma_window=ma_window,
        p_threshold=p_threshold,
        **kwargs
    )
