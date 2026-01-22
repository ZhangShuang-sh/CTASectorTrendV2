#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Price Volume Pattern Factor

价量模式匹配因子

基于广发证券《价量模式匹配股指期货交易策略》（另类交易策略系列之二十五）

核心算法：动态时间规整（DTW）
- 从历史数据中寻找与当前行情最相似的片段
- 根据相似片段的后续走势预测未来收益
- 同时考虑价格和成交量进行模式匹配

原报告业绩（L=11, 2013年以来）：
- 年化收益率：35.5%
- 最大回撤：-12.8%
- 胜率：49.7%
- 盈亏比：1.29

Source: V1 factors/time_series/price_volume_pattern.py (100% logic preserved)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from core.factors.time_series.base import TimeSeriesFactorBase
from core.factors.registry import register


def _dtw_distance_pv(seq1: np.ndarray, seq2: np.ndarray,
                     var_p: float, var_v: float) -> float:
    """
    计算两个序列的DTW距离（末端对齐版本）

    Args:
        seq1: 当前模式 (L, 2) - [normalized_price, normalized_volume]
        seq2: 历史模式 (L, 2) - [normalized_price, normalized_volume]
        var_p: 价格方差
        var_v: 成交量方差

    Returns:
        float: DTW距离
    """
    m, n = len(seq1), len(seq2)

    # 初始化距离矩阵
    D = np.full((m + 1, n + 1), np.inf)
    D[m, n] = 0.0

    # 从末端向前计算（末端对齐）
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            # 点距离（标准化）
            d_price = (seq1[i, 0] - seq2[j, 0]) ** 2 / (var_p + 1e-8)
            d_vol = (seq1[i, 1] - seq2[j, 1]) ** 2 / (var_v + 1e-8)
            d_ij = np.sqrt(d_price + d_vol)

            # 递推
            D[i, j] = d_ij + min(D[i+1, j], D[i, j+1], D[i+1, j+1])

    # 返回最小距离（允许子序列匹配）
    return min(D[0, :].min(), D[:, 0].min())


def _dtw_distance_price_only(seq1: np.ndarray, seq2: np.ndarray,
                              var_p: float) -> float:
    """
    计算两个序列的DTW距离（仅价格）
    """
    m, n = len(seq1), len(seq2)
    D = np.full((m + 1, n + 1), np.inf)
    D[m, n] = 0.0

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            d_ij = np.abs(seq1[i] - seq2[j]) / np.sqrt(var_p + 1e-8)
            D[i, j] = d_ij + min(D[i+1, j], D[i, j+1], D[i+1, j+1])

    return min(D[0, :].min(), D[:, 0].min())


@register('PriceVolumePatternFactor')
class PriceVolumePatternFactor(TimeSeriesFactorBase):
    """
    价量模式匹配因子

    基于动态时间规整(DTW)算法，从历史数据中寻找与当前行情
    最相似的片段，根据历史片段的后续走势预测未来收益。

    信号输出：
    - +1: 预测上涨，做多
    - -1: 预测下跌，做空
    - 0: 无信号（数据不足）

    参数：
    - pattern_length: 模式长度L（默认11）
    - top_k: 选取的相似片段数量（默认5）
    - min_history: 最小历史样本数量（默认50）
    - use_volume: 是否使用成交量（默认True）
    - volume_ma_period: 成交量移动平均周期（默认50）
    """

    def __init__(
        self,
        name: str = None,
        pattern_length: int = 11,
        top_k: int = 5,
        min_history: int = 50,
        use_volume: bool = True,
        volume_ma_period: int = 50,
        window: int = None
    ):
        """
        Args:
            name: 因子名称
            pattern_length: 模式长度L
            top_k: 选取的相似片段数量
            min_history: 最小历史样本数量
            use_volume: 是否使用成交量
            volume_ma_period: 成交量移动平均周期
            window: 回看窗口
        """
        if name is None:
            vol_str = "PV" if use_volume else "P"
            name = f"PriceVolumePattern_{vol_str}_L{pattern_length}_K{top_k}"

        if window is None:
            window = max(pattern_length + min_history, 100)

        super().__init__(name=name, window=window)

        self.pattern_length = pattern_length
        self.top_k = top_k
        self.min_history = min_history
        self.use_volume = use_volume
        self.volume_ma_period = volume_ma_period

        self._params.update({
            'pattern_length': pattern_length,
            'top_k': top_k,
            'min_history': min_history,
            'use_volume': use_volume,
            'volume_ma_period': volume_ma_period
        })

    def _get_prices(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """获取收盘价序列"""
        for col in ['close', 'S_DQ_CLOSE', 'CLOSE']:
            if col in data.columns:
                return data[col]
        return None

    def _get_volumes(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """获取成交量序列"""
        for col in ['volume', 'S_DQ_VOLUME', 'VOLUME']:
            if col in data.columns:
                return data[col]
        return None

    def _normalize_price(self, prices: np.ndarray) -> np.ndarray:
        """价格标准化：除以最新收盘价"""
        if len(prices) == 0 or prices[-1] == 0:
            return prices
        return prices / prices[-1]

    def _normalize_volume(self, volumes: np.ndarray,
                         volume_ma: np.ndarray) -> np.ndarray:
        """成交量标准化：除以移动平均"""
        result = np.zeros_like(volumes, dtype=float)
        valid_mask = volume_ma > 0
        result[valid_mask] = volumes[valid_mask] / volume_ma[valid_mask]
        return result

    def _create_pattern(self, prices: np.ndarray, volumes: np.ndarray,
                       idx: int, length: int) -> Optional[np.ndarray]:
        """创建价量模式"""
        if idx < length - 1:
            return None

        start_idx = idx - length + 1
        price_pattern = prices[start_idx:idx + 1].copy()

        # 对价格进行片段内标准化
        if price_pattern[-1] != 0:
            price_pattern = price_pattern / price_pattern[-1]

        if self.use_volume:
            volume_pattern = volumes[start_idx:idx + 1]
            return np.column_stack([price_pattern, volume_pattern])
        else:
            return price_pattern.reshape(-1, 1)

    def _compute_dtw_distance(self, pattern1: np.ndarray,
                              pattern2: np.ndarray) -> float:
        """计算两个模式的DTW距离"""
        var_p = np.var(pattern1[:, 0]) + 1e-8
        var_v = np.var(pattern1[:, 1]) + 1e-8 if pattern1.shape[1] > 1 else 1.0

        if self.use_volume and pattern1.shape[1] > 1:
            return _dtw_distance_pv(pattern1, pattern2, var_p, var_v)
        else:
            return _dtw_distance_price_only(pattern1[:, 0], pattern2[:, 0], var_p)

    def _find_similar_patterns(
        self,
        current_pattern: np.ndarray,
        historical_patterns: List[Tuple[np.ndarray, float]],
        top_k: int
    ) -> List[Tuple[float, float]]:
        """寻找最相似的历史模式"""
        distances = []

        for hist_pattern, next_return in historical_patterns:
            dist = self._compute_dtw_distance(current_pattern, hist_pattern)
            distances.append((dist, next_return))

        distances.sort(key=lambda x: x[0])
        return distances[:top_k]

    def _predict_return(self, similar_patterns: List[Tuple[float, float]]) -> float:
        """基于相似模式加权预测收益"""
        if not similar_patterns:
            return 0.0

        total_weight = 0.0
        weighted_return = 0.0

        for dist, ret in similar_patterns:
            if dist > 0:
                weight = 1.0 / dist
                weighted_return += weight * ret
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_return / total_weight

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算当前信号

        Args:
            data: 必须包含 'close' 列，可选 'volume' 列

        Returns:
            float: 信号值 (-1, 0, 1)
        """
        signals = self.calculate_series(data)
        if len(signals) == 0:
            return 0.0
        return float(signals.iloc[-1])

    def calculate_series(self, data: pd.DataFrame) -> pd.Series:
        """
        计算完整信号序列

        Args:
            data: 必须包含 'close' 列，可选 'volume' 列

        Returns:
            pd.Series: 信号序列 (-1, 0, 1)
        """
        prices = self._get_prices(data)
        volumes = self._get_volumes(data) if self.use_volume else None

        if prices is None or len(prices) < self.pattern_length + self.min_history:
            return pd.Series(dtype=float)

        # 转换为numpy数组
        price_arr = prices.values.astype(float)
        volume_arr = volumes.values.astype(float) if volumes is not None else np.ones_like(price_arr)

        # 计算成交量移动平均
        volume_ma = pd.Series(volume_arr).rolling(
            window=self.volume_ma_period, min_periods=1
        ).mean().values

        # 标准化成交量
        norm_volume = self._normalize_volume(volume_arr, volume_ma)

        # 计算收益率
        returns = np.zeros(len(price_arr))
        returns[1:] = (price_arr[1:] - price_arr[:-1]) / price_arr[:-1]

        # 初始化信号序列
        n = len(price_arr)
        signals = pd.Series(index=prices.index, dtype=float)
        signals[:] = 0.0

        # 从有足够历史数据的位置开始
        start_idx = self.pattern_length + self.min_history

        for t in range(start_idx, n):
            # 创建当前模式
            current_pattern = self._create_pattern(
                price_arr[:t+1], norm_volume[:t+1], t, self.pattern_length
            )

            if current_pattern is None:
                continue

            # 构建历史模式库
            historical_patterns = []
            for hist_idx in range(self.pattern_length - 1, t - 1):
                hist_pattern = self._create_pattern(
                    price_arr[:hist_idx+1], norm_volume[:hist_idx+1],
                    hist_idx, self.pattern_length
                )
                if hist_pattern is not None and hist_idx + 1 < len(returns):
                    next_return = returns[hist_idx + 1]
                    historical_patterns.append((hist_pattern, next_return))

            if len(historical_patterns) < self.top_k:
                continue

            # 寻找相似模式
            similar_patterns = self._find_similar_patterns(
                current_pattern, historical_patterns, self.top_k
            )

            # 预测收益
            predicted_return = self._predict_return(similar_patterns)

            # 生成信号
            if predicted_return > 0:
                signals.iloc[t] = 1.0
            elif predicted_return < 0:
                signals.iloc[t] = -1.0
            else:
                signals.iloc[t] = 0.0

        return signals


# 便捷工厂函数
def create_pv_pattern_factor(
    pattern_length: int = 11,
    top_k: int = 5,
    use_volume: bool = True
) -> PriceVolumePatternFactor:
    """创建价量模式匹配因子（推荐配置）"""
    return PriceVolumePatternFactor(
        pattern_length=pattern_length,
        top_k=top_k,
        use_volume=use_volume
    )


def create_price_only_pattern_factor(
    pattern_length: int = 11,
    top_k: int = 5
) -> PriceVolumePatternFactor:
    """创建纯价格模式匹配因子（对照组）"""
    return PriceVolumePatternFactor(
        pattern_length=pattern_length,
        top_k=top_k,
        use_volume=False
    )
