#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Signal Normalizer

Provides signal normalization methods:
- zscore_clip: Z-score normalization with clipping to [-1, 1]
- minmax: Min-max normalization to [0, 1] or [-1, 1]
- rank: Rank normalization to [0, 1]
- none: No normalization

Each factor can specify its preferred normalization method.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from enum import Enum


class NormalizationMethod(Enum):
    """正则化方法枚举"""
    ZSCORE_CLIP = "zscore_clip"
    MINMAX = "minmax"
    RANK = "rank"
    NONE = "none"


class SignalNormalizer:
    """
    信号正则化器

    功能:
    1. 将因子原始值正则化到标准范围
    2. 支持多种正则化方法
    3. 支持时序和截面两种模式
    """

    def __init__(
        self,
        default_method: str = "zscore_clip",
        clip_range: tuple = (-1.0, 1.0),
        zscore_clip_std: float = 3.0
    ):
        """
        Args:
            default_method: 默认正则化方法
            clip_range: 裁剪范围
            zscore_clip_std: Z-score 裁剪标准差倍数
        """
        self.default_method = default_method
        self.clip_range = clip_range
        self.zscore_clip_std = zscore_clip_std

    def normalize(
        self,
        values: Union[float, np.ndarray, pd.Series, Dict],
        method: str = None,
        **kwargs
    ) -> Union[float, np.ndarray, pd.Series, Dict]:
        """
        正则化信号值

        Args:
            values: 原始信号值 (标量、数组、Series或Dict)
            method: 正则化方法 (None 使用默认方法)
            **kwargs: 额外参数

        Returns:
            正则化后的信号值
        """
        if method is None:
            method = self.default_method

        # 标量处理
        if isinstance(values, (int, float)):
            return self._normalize_scalar(values, method)

        # 字典处理 (配对因子的输出)
        if isinstance(values, dict):
            return {k: self.normalize(v, method, **kwargs) for k, v in values.items()}

        # 数组/Series 处理
        if isinstance(values, pd.Series):
            return self._normalize_series(values, method, **kwargs)

        if isinstance(values, np.ndarray):
            return self._normalize_array(values, method, **kwargs)

        return values

    def _normalize_scalar(self, value: float, method: str) -> float:
        """正则化标量值"""
        if pd.isna(value) or np.isinf(value):
            return 0.0

        if method == "none":
            return value

        # 标量无法做 zscore/minmax/rank，只能裁剪
        return float(np.clip(value, self.clip_range[0], self.clip_range[1]))

    def _normalize_series(
        self,
        series: pd.Series,
        method: str,
        **kwargs
    ) -> pd.Series:
        """正则化 Series"""
        if series.empty:
            return series

        if method == "zscore_clip":
            return self._zscore_clip_series(series, **kwargs)
        elif method == "minmax":
            return self._minmax_series(series, **kwargs)
        elif method == "rank":
            return self._rank_series(series, **kwargs)
        elif method == "none":
            return series
        else:
            # 默认 zscore_clip
            return self._zscore_clip_series(series, **kwargs)

    def _normalize_array(
        self,
        arr: np.ndarray,
        method: str,
        **kwargs
    ) -> np.ndarray:
        """正则化数组"""
        if len(arr) == 0:
            return arr

        if method == "zscore_clip":
            return self._zscore_clip_array(arr, **kwargs)
        elif method == "minmax":
            return self._minmax_array(arr, **kwargs)
        elif method == "rank":
            return self._rank_array(arr, **kwargs)
        elif method == "none":
            return arr
        else:
            return self._zscore_clip_array(arr, **kwargs)

    # ========== Z-Score Clip ==========

    def _zscore_clip_series(
        self,
        series: pd.Series,
        rolling_window: int = None,
        **kwargs
    ) -> pd.Series:
        """
        Z-Score 正则化 + 裁剪

        z = (x - mean) / std
        然后裁剪到 [-clip_std, clip_std]
        最后缩放到 clip_range
        """
        if rolling_window:
            mean = series.rolling(rolling_window).mean()
            std = series.rolling(rolling_window).std()
        else:
            mean = series.mean()
            std = series.std()

        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)

        zscore = (series - mean) / std

        # 裁剪到 [-clip_std, clip_std]
        zscore_clipped = zscore.clip(-self.zscore_clip_std, self.zscore_clip_std)

        # 缩放到 clip_range
        scaled = zscore_clipped / self.zscore_clip_std  # -> [-1, 1]

        if self.clip_range != (-1.0, 1.0):
            # 线性变换到目标范围
            range_size = self.clip_range[1] - self.clip_range[0]
            scaled = scaled * (range_size / 2) + (self.clip_range[0] + self.clip_range[1]) / 2

        return scaled

    def _zscore_clip_array(
        self,
        arr: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Z-Score 正则化数组"""
        valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
        valid_arr = arr[valid_mask]

        if len(valid_arr) == 0:
            return np.zeros_like(arr)

        mean = np.mean(valid_arr)
        std = np.std(valid_arr)

        if std == 0:
            return np.zeros_like(arr)

        zscore = (arr - mean) / std
        zscore_clipped = np.clip(zscore, -self.zscore_clip_std, self.zscore_clip_std)
        scaled = zscore_clipped / self.zscore_clip_std

        return scaled

    # ========== Min-Max ==========

    def _minmax_series(
        self,
        series: pd.Series,
        target_range: tuple = None,
        **kwargs
    ) -> pd.Series:
        """
        Min-Max 正则化

        x_norm = (x - min) / (max - min)
        """
        if target_range is None:
            target_range = self.clip_range

        min_val = series.min()
        max_val = series.max()

        if max_val == min_val:
            return pd.Series(
                (target_range[0] + target_range[1]) / 2,
                index=series.index
            )

        normalized = (series - min_val) / (max_val - min_val)

        # 缩放到目标范围
        range_size = target_range[1] - target_range[0]
        return normalized * range_size + target_range[0]

    def _minmax_array(
        self,
        arr: np.ndarray,
        target_range: tuple = None,
        **kwargs
    ) -> np.ndarray:
        """Min-Max 正则化数组"""
        if target_range is None:
            target_range = self.clip_range

        valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
        valid_arr = arr[valid_mask]

        if len(valid_arr) == 0:
            return np.zeros_like(arr)

        min_val = np.min(valid_arr)
        max_val = np.max(valid_arr)

        if max_val == min_val:
            return np.full_like(arr, (target_range[0] + target_range[1]) / 2)

        normalized = (arr - min_val) / (max_val - min_val)
        range_size = target_range[1] - target_range[0]

        return normalized * range_size + target_range[0]

    # ========== Rank ==========

    def _rank_series(
        self,
        series: pd.Series,
        to_symmetric: bool = True,
        **kwargs
    ) -> pd.Series:
        """
        排名正则化

        使用百分位排名，结果在 [0, 1]
        如果 to_symmetric=True，则转换到 [-1, 1]
        """
        ranked = series.rank(pct=True)

        if to_symmetric:
            # [0, 1] -> [-1, 1]
            ranked = ranked * 2 - 1

        return ranked

    def _rank_array(
        self,
        arr: np.ndarray,
        to_symmetric: bool = True,
        **kwargs
    ) -> np.ndarray:
        """排名正则化数组"""
        from scipy.stats import rankdata

        valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
        result = np.zeros_like(arr)

        if np.sum(valid_mask) == 0:
            return result

        valid_arr = arr[valid_mask]
        ranks = rankdata(valid_arr, method='average')
        n = len(valid_arr)

        # 转换为百分位
        percentile_ranks = (ranks - 1) / (n - 1) if n > 1 else np.zeros_like(ranks)

        result[valid_mask] = percentile_ranks

        if to_symmetric:
            result = result * 2 - 1

        return result

    # ========== Cross-Sectional Normalization ==========

    def normalize_cross_section(
        self,
        factor_values: Dict[str, float],
        method: str = None
    ) -> Dict[str, float]:
        """
        截面正则化

        对一组资产的因子值进行截面正则化

        Args:
            factor_values: {asset: factor_value}
            method: 正则化方法

        Returns:
            {asset: normalized_value}
        """
        if not factor_values:
            return {}

        if method is None:
            method = self.default_method

        series = pd.Series(factor_values)
        normalized = self._normalize_series(series, method)

        return normalized.to_dict()

    def __repr__(self) -> str:
        return f"SignalNormalizer(method={self.default_method}, clip={self.clip_range})"


# 便捷函数
def create_normalizer(
    method: str = "zscore_clip",
    clip_range: tuple = (-1.0, 1.0)
) -> SignalNormalizer:
    """创建信号正则化器"""
    return SignalNormalizer(
        default_method=method,
        clip_range=clip_range
    )


def zscore_normalize(values: Union[pd.Series, np.ndarray], clip_std: float = 3.0) -> Union[pd.Series, np.ndarray]:
    """便捷函数：Z-score 正则化"""
    normalizer = SignalNormalizer(zscore_clip_std=clip_std)
    return normalizer.normalize(values, method="zscore_clip")


def rank_normalize(values: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """便捷函数：排名正则化"""
    normalizer = SignalNormalizer()
    return normalizer.normalize(values, method="rank")
