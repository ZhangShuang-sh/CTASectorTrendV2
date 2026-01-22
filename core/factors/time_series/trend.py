#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Trend Time Series Factors

Factors:
- HurstExponent: Hurst exponent for trend strength detection
- EMDTrend: EMD trend efficiency factor

Source: V1 factors/time_series/trend.py (100% logic preserved)
"""

import numpy as np
import pandas as pd

from core.factors.time_series.base import TimeSeriesFactorBase
from core.factors.registry import register


@register('HurstExponent')
class HurstExponent(TimeSeriesFactorBase):
    """
    Hurst Exponent (赫斯特指数)

    来源: 报告08《如何有效使用赫斯特指数做趋势增强》

    衡量价格序列的长期记忆性:
    - H > 0.5: 存在趋势（值越大趋势越强）
    - H < 0.5: 均值回归（震荡）
    - H = 0.5: 随机游走

    使用 R/S 分析法计算。
    """

    def __init__(self, name: str = None, window: int = 100, scale: float = 2.0, return_raw: bool = False):
        """
        Args:
            name: 因子名称
            window: 回看窗口长度
            scale: 信号放大系数（当 return_raw=False 时使用）
            return_raw: 是否返回原始 H 值，默认返回 (H - 0.5) * scale
        """
        if name is None:
            name = f"Hurst_{window}"

        super().__init__(name=name, window=window)
        self.scale = scale
        self.return_raw = return_raw

        self._params.update({
            'scale': scale,
            'return_raw': return_raw
        })

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算 Hurst 指数。

        Args:
            data: 必须包含 'S_DQ_CLOSE' 或 'close' 列

        Returns:
            float: 信号强度值
        """
        if not self.validate_inputs(data):
            return 0.0

        # 支持两种列名格式
        close_col = 'S_DQ_CLOSE' if 'S_DQ_CLOSE' in data.columns else 'close'
        prices = data[close_col].values[-self.window:]

        # 计算对数收益率
        log_returns = np.diff(np.log(prices))

        if len(log_returns) < 10:
            return 0.0

        # R/S 分析法计算 Hurst 指数
        h = self._rs_analysis(log_returns)

        if self.return_raw:
            return float(h)
        else:
            # 转换为信号: (H - 0.5) * scale
            # H > 0.5 -> 正向趋势信号
            # H < 0.5 -> 负向（震荡/反转）信号
            return self._clip_signal((h - 0.5) * self.scale)

    def _rs_analysis(self, returns: np.ndarray) -> float:
        """
        R/S 分析法计算 Hurst 指数

        通过多个子区间计算 log(R/S) vs log(n) 的斜率
        """
        n = len(returns)

        # 定义不同的子区间长度
        # 使用 2 的幂次作为区间划分
        max_k = int(np.floor(np.log2(n)))
        if max_k < 2:
            return 0.5

        ns = []
        rs_values = []

        for k in range(2, max_k + 1):
            size = 2 ** k
            if size > n:
                break

            # 计算该区间长度下的 R/S 值
            rs = self._compute_rs(returns, size)
            if rs > 0:
                ns.append(size)
                rs_values.append(rs)

        if len(ns) < 2:
            return 0.5

        # 线性回归: log(R/S) = H * log(n) + c
        log_ns = np.log(ns)
        log_rs = np.log(rs_values)

        # 最小二乘法求斜率
        n_pts = len(log_ns)
        sum_x = np.sum(log_ns)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_ns * log_rs)
        sum_x2 = np.sum(log_ns ** 2)

        denominator = n_pts * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.5

        h = (n_pts * sum_xy - sum_x * sum_y) / denominator

        # 限制 H 值在合理范围内 [0, 1]
        return np.clip(h, 0.0, 1.0)

    def _compute_rs(self, returns: np.ndarray, size: int) -> float:
        """
        计算给定区间长度的 R/S 值
        """
        n = len(returns)
        num_segments = n // size

        if num_segments == 0:
            return 0.0

        rs_list = []

        for i in range(num_segments):
            segment = returns[i * size:(i + 1) * size]

            # 计算均值离差
            mean_return = np.mean(segment)
            deviations = segment - mean_return

            # 累积离差
            cumulative_dev = np.cumsum(deviations)

            # 极差 R
            r = np.max(cumulative_dev) - np.min(cumulative_dev)

            # 标准差 S
            s = np.std(segment, ddof=1)

            if s > 1e-10:
                rs_list.append(r / s)

        if len(rs_list) == 0:
            return 0.0

        return np.mean(rs_list)


@register('EMDTrend')
class EMDTrend(TimeSeriesFactorBase):
    """
    EMD Trend Efficiency (经验模态分解趋势度)

    来源: 报告14/39《经验模态分解下的日内趋势交易》

    利用 EMD 将价格分解为多个 IMF 和残差项:
    - 低频 IMF + 残差 = 趋势项 (Trend)
    - 高频 IMF = 噪声项 (Noise)

    输出: 趋势能量占比，介于 0~1 之间

    注意: 完整 EMD 实现需要 PyEMD 库 (pip install EMD-signal)
    本实现提供简化版，使用移动平均近似分解
    """

    def __init__(self, name: str = None, window: int = 60, noise_imfs: int = 2, use_simple: bool = True):
        """
        Args:
            name: 因子名称
            window: 回看窗口长度
            noise_imfs: 定义前几个 IMF 为噪声（默认前 2 个）
            use_simple: 是否使用简化版（移动平均近似），默认 True
                       设为 False 需要安装 PyEMD 库
        """
        if name is None:
            name = f"EMDTrend_{window}"

        super().__init__(name=name, window=window)
        self.noise_imfs = noise_imfs
        self.use_simple = use_simple

        self._params.update({
            'noise_imfs': noise_imfs,
            'use_simple': use_simple
        })

    def calculate(self, data: pd.DataFrame, **kwargs) -> float:
        """
        计算 EMD 趋势效率。

        Args:
            data: 必须包含 'S_DQ_CLOSE' 或 'close' 列

        Returns:
            float: 趋势能量占比，范围 [0, 1]
                   值越高表示趋势行情概率越大
        """
        if not self.validate_inputs(data):
            return 0.5

        close_col = 'S_DQ_CLOSE' if 'S_DQ_CLOSE' in data.columns else 'close'
        prices = data[close_col].values[-self.window:]

        if self.use_simple:
            return self._compute_simple(prices)
        else:
            return self._compute_emd(prices)

    def _compute_simple(self, prices: np.ndarray) -> float:
        """
        简化版 EMD 趋势度计算

        使用多尺度移动平均近似分解:
        - 短期 MA 变化 -> 噪声
        - 长期 MA 变化 -> 趋势
        """
        n = len(prices)

        # 定义多个尺度
        short_windows = [3, 5]  # 高频/噪声
        long_windows = [10, 20, n // 2]  # 低频/趋势

        # 计算各尺度的能量（方差）
        noise_energy = 0.0
        trend_energy = 0.0

        # 噪声: 短期波动
        for w in short_windows:
            if w < n:
                ma = self._moving_average(prices, w)
                residual = prices[w - 1:] - ma
                noise_energy += np.var(residual)

        # 趋势: 长期变化
        for w in long_windows:
            if w < n and w > 1:
                ma = self._moving_average(prices, w)
                trend_energy += np.var(ma)

        total_energy = noise_energy + trend_energy

        if total_energy < 1e-10:
            return 0.5

        # 趋势能量占比
        ratio = trend_energy / total_energy

        return float(np.clip(ratio, 0.0, 1.0))

    def _compute_emd(self, prices: np.ndarray) -> float:
        """
        使用 PyEMD 进行完整 EMD 分解

        依赖: pip install EMD-signal
        """
        try:
            from PyEMD import EMD
        except ImportError:
            # 如果没有 PyEMD，回退到简化版
            return self._compute_simple(prices)

        emd = EMD()

        try:
            imfs = emd.emd(prices)
        except Exception:
            return self._compute_simple(prices)

        if imfs is None or len(imfs) == 0:
            return 0.5

        # 计算能量
        noise_energy = 0.0
        trend_energy = 0.0

        for i, imf in enumerate(imfs):
            energy = np.sum(imf ** 2)
            if i < self.noise_imfs:
                noise_energy += energy
            else:
                trend_energy += energy

        total_energy = noise_energy + trend_energy

        if total_energy < 1e-10:
            return 0.5

        return float(trend_energy / total_energy)

    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """简单移动平均"""
        return np.convolve(data, np.ones(window) / window, mode='valid')
