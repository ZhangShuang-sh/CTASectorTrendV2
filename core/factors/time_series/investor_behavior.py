#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于投资者行为的趋势因子

来源: 【中信期货金融工程】期货多因子系列之九：基于投资者行为的趋势因子——专题报告20241024

理论背景:
- 基于 Moskowitz (2021) 的投资者行为理论
- 使用波动率度量投资者对合约的"了解程度"
- 高波动 -> 不了解 -> 趋势延续 -> 跟随动量
- 低波动 -> 了解 -> 均值回归 -> 反转动量

核心逻辑：
这是一个统一因子，根据波动率状态自动切换动量/反转模式：
- 当波动率 > 历史分位数阈值：跟随动量（趋势延续）
- 当波动率 <= 历史分位数阈值：反转动量（均值回归）
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.factors.time_series.base import TimeSeriesFactorBase


class InvestorBehaviorTrendFactor(TimeSeriesFactorBase):
    """
    基于投资者行为的趋势因子（统一版本）

    核心逻辑：
    1. 计算动量信号 (可配置回看期)
    2. 计算多窗口滚动波动率（可选遗忘机制）
    3. 根据波动率分位数决定策略：
       - 高波动 (vol > quantile) → 跟随动量（趋势延续）
       - 低波动 (vol <= quantile) → 反转动量（均值回归）
    4. 应用均值方差增强提升稳定性

    最优参数（短期版本）:
    - mom_w=1, revs_w=3, revs_mean_w=10, revs_thr=0.8
    - mean_w=10, std_w=5, weight=0.7, use_forget=False

    最优参数（长期版本）:
    - mom_w=60, revs_w=10, revs_mean_w=10, revs_thr=0.8
    - mean_w=5, std_w=5, weight=0.2, use_forget=True
    """

    def __init__(self,
                 mom_w: int = 1,
                 revs_w: int = 3,
                 revs_mean_w: int = 10,
                 revs_thr: float = 0.8,
                 mean_w: int = 10,
                 std_w: int = 5,
                 weight: float = 0.7,
                 quantile_window: int = 60,
                 use_forget: bool = False,
                 use_meanstd_enhance: bool = True):
        """
        Args:
            mom_w: 动量回看期（天），短期用1，长期用60
            revs_w: 短期波动率窗口
            revs_mean_w: 波动率均值窗口
            revs_thr: 反转分位数门槛 (0.8 = 80%分位数)
            mean_w: 均值增强窗口
            std_w: 方差增强窗口
            weight: 均值权重 (1-weight为方差权重)
            quantile_window: 分位数计算窗口（时序测试用）
            use_forget: 是否使用遗忘机制（长期版本建议开启）
            use_meanstd_enhance: 是否使用均值方差增强
        """
        super().__init__(
            name=f"InvBehaviorTrend_{mom_w}_{revs_w}_{revs_mean_w}",
            window=max(mom_w + revs_w + revs_mean_w + mean_w + std_w + quantile_window, 150)
        )
        self.mom_w = mom_w
        self.revs_w = revs_w
        self.revs_mean_w = revs_mean_w
        self.revs_thr = revs_thr
        self.mean_w = mean_w
        self.std_w = std_w
        self.weight = weight
        self.quantile_window = quantile_window
        self.use_forget = use_forget
        self.use_meanstd_enhance = use_meanstd_enhance

    def set_params(self, **kwargs) -> 'InvestorBehaviorTrendFactor':
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _weighted_mean(self, series: pd.Series, window: int) -> pd.Series:
        """
        加权平均（遗忘机制）

        近期数据权重更高，远期数据权重更低
        权重序列: [1, 2, 3, ..., window] / sum
        """
        weights = np.arange(1, window + 1, dtype=float)
        weights = weights / weights.sum()

        def apply_weights(x):
            if len(x) < window:
                return np.nan
            return np.sum(x[-window:] * weights)

        return series.rolling(window).apply(apply_weights, raw=True)

    def calculate(self, data: pd.DataFrame) -> float:
        """
        计算投资者行为趋势因子值

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            float: 信号值，正值看多，负值看空
        """
        if len(data) < self.window:
            return 0.0

        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return 0.0

        prices = data[close_col]

        # 计算收益率
        returns = prices.pct_change().dropna()
        if len(returns) < self.window - 1:
            return 0.0

        # 1. 计算动量
        momentum = returns.rolling(self.mom_w).mean()

        # 2. 计算多窗口滚动波动率
        short_vol = returns.rolling(self.revs_w).std()

        if self.use_forget:
            # 使用遗忘机制（加权平均，近期权重高）
            vol_mean = self._weighted_mean(short_vol, self.revs_mean_w)
        else:
            # 简单平均
            vol_mean = short_vol.rolling(self.revs_mean_w).mean()

        # 3. 计算历史滚动分位数（时序测试）
        vol_quantile = vol_mean.rolling(self.quantile_window).quantile(self.revs_thr)

        # 4. 根据波动率决定策略方向
        # 核心逻辑：
        # - 高波动 (vol > quantile) → 趋势延续 → 跟随动量
        # - 低波动 (vol <= quantile) → 均值回归 → 反转动量
        base_factor = momentum.copy()
        mask_low_vol = vol_mean <= vol_quantile
        base_factor[mask_low_vol] = -momentum[mask_low_vol]

        if not self.use_meanstd_enhance:
            result = base_factor.iloc[-1]
            return float(result) if not pd.isna(result) else 0.0

        # 5. 均值方差增强
        factor_mean = base_factor.rolling(self.mean_w).mean()
        factor_std = base_factor.rolling(self.std_w).std()

        # 标准化（避免量纲差异）
        mean_scale = factor_mean.abs().rolling(20).mean()
        std_scale = factor_std.rolling(20).mean()

        factor_mean_norm = factor_mean / mean_scale.replace(0, np.nan)
        factor_std_norm = factor_std / std_scale.replace(0, np.nan)

        # 加权组合
        enhanced = self.weight * factor_mean_norm + (1 - self.weight) * factor_std_norm

        result = enhanced.iloc[-1]
        if pd.isna(result):
            return 0.0

        # 限制输出范围
        return float(np.clip(result, -3.0, 3.0))

    def compute_series(self, data: pd.DataFrame) -> pd.Series:
        """
        计算完整的因子序列（用于回测）

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            pd.Series: 因子值序列
        """
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return pd.Series(dtype=float)

        prices = data[close_col]

        # 计算收益率
        returns = prices.pct_change()

        # 1. 计算动量
        momentum = returns.rolling(self.mom_w).mean()

        # 2. 计算多窗口滚动波动率
        short_vol = returns.rolling(self.revs_w).std()

        if self.use_forget:
            vol_mean = self._weighted_mean(short_vol, self.revs_mean_w)
        else:
            vol_mean = short_vol.rolling(self.revs_mean_w).mean()

        # 3. 计算历史滚动分位数
        vol_quantile = vol_mean.rolling(self.quantile_window).quantile(self.revs_thr)

        # 4. 根据波动率决定策略方向
        base_factor = momentum.copy()
        mask_low_vol = vol_mean <= vol_quantile
        base_factor[mask_low_vol] = -momentum[mask_low_vol]

        if not self.use_meanstd_enhance:
            return base_factor

        # 5. 均值方差增强
        factor_mean = base_factor.rolling(self.mean_w).mean()
        factor_std = base_factor.rolling(self.std_w).std()

        mean_scale = factor_mean.abs().rolling(20).mean()
        std_scale = factor_std.rolling(20).mean()

        factor_mean_norm = factor_mean / mean_scale.replace(0, np.nan)
        factor_std_norm = factor_std / std_scale.replace(0, np.nan)

        enhanced = self.weight * factor_mean_norm + (1 - self.weight) * factor_std_norm

        return enhanced.clip(-3.0, 3.0)


# 便捷工厂函数
def create_short_term_factor(**kwargs) -> InvestorBehaviorTrendFactor:
    """
    创建短期版本因子（适合捕捉短期趋势）

    默认参数: mom_w=1, revs_w=3, weight=0.7, use_forget=False
    """
    defaults = {
        'mom_w': 1,
        'revs_w': 3,
        'revs_mean_w': 10,
        'revs_thr': 0.8,
        'mean_w': 10,
        'std_w': 5,
        'weight': 0.7,
        'use_forget': False,
        'use_meanstd_enhance': True
    }
    defaults.update(kwargs)
    return InvestorBehaviorTrendFactor(**defaults)


def create_long_term_factor(**kwargs) -> InvestorBehaviorTrendFactor:
    """
    创建长期版本因子（适合捕捉长期趋势）

    默认参数: mom_w=60, revs_w=10, weight=0.2, use_forget=True
    """
    defaults = {
        'mom_w': 60,
        'revs_w': 10,
        'revs_mean_w': 10,
        'revs_thr': 0.8,
        'mean_w': 5,
        'std_w': 5,
        'weight': 0.2,
        'use_forget': True,
        'use_meanstd_enhance': True
    }
    defaults.update(kwargs)
    return InvestorBehaviorTrendFactor(**defaults)


# 保留旧类名兼容性（指向统一因子）
class InvestorBehaviorMomentum(InvestorBehaviorTrendFactor):
    """短期版本 (mom_w=1) - 兼容旧代码"""
    def __init__(self, **kwargs):
        defaults = {
            'mom_w': 1,
            'revs_w': 3,
            'revs_mean_w': 10,
            'mean_w': 10,
            'std_w': 5,
            'weight': 0.7,
            'use_forget': False
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class InvestorBehaviorReversal(InvestorBehaviorTrendFactor):
    """长期版本 (mom_w=60) - 兼容旧代码"""
    def __init__(self, **kwargs):
        defaults = {
            'mom_w': 60,
            'revs_w': 10,
            'revs_mean_w': 10,
            'mean_w': 5,
            'std_w': 5,
            'weight': 0.2,
            'use_forget': True
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
