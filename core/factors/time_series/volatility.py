#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序波动率/均值回归因子模块

包含:
- KalmanFilterDeviation: 卡尔曼滤波偏离度
- KalmanTrendFollower: 卡尔曼趋势跟踪因子

来源: 报告01/03《卡尔曼滤波最优性原理》
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any

from core.factors.time_series.base import TimeSeriesFactorBase


class KalmanFilterDeviation(TimeSeriesFactorBase):
    """
    Kalman Filter Deviation (卡尔曼滤波偏离度)

    利用卡尔曼滤波估计价格的"真实状态"，计算当前价格与估计值的偏离。

    状态方程: x(t) = x(t-1) + w  (假设真实价格是平稳过程)
    观测方程: z(t) = x(t) + v    (观测价格含噪声)

    输出: Z-Score 标准化后的偏离度 (Spread)
    绝对值 > 2 提示反转机会
    """

    def __init__(self, window: int = 60, process_noise: float = 1e-5,
                 measurement_noise: float = 1e-3, zscore_window: int = 20):
        """
        Args:
            window: 用于计算的回看窗口长度
            process_noise: 过程噪声方差 Q
            measurement_noise: 观测噪声方差 R
            zscore_window: 计算 Z-Score 的窗口长度
        """
        super().__init__(name=f"KalmanDev_{window}", window=window)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.zscore_window = zscore_window

    def set_params(self, **kwargs) -> 'KalmanFilterDeviation':
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def calculate(self, data: pd.DataFrame) -> float:
        """
        计算卡尔曼滤波偏离度。

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            float: Z-Score 标准化后的偏离度
                   正值: 价格高于均衡状态，可能回落
                   负值: 价格低于均衡状态，可能反弹
        """
        if len(data) < self.window:
            return 0.0

        # 获取价格列
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return 0.0

        prices = data[close_col].values[-self.window:]

        # 运行卡尔曼滤波
        filtered_prices, spreads = self._kalman_filter(prices)

        if len(spreads) < self.zscore_window:
            return 0.0

        # 计算 Z-Score
        recent_spreads = spreads[-self.zscore_window:]
        mean_spread = np.mean(recent_spreads)
        std_spread = np.std(recent_spreads)

        if std_spread < 1e-10:
            return 0.0

        current_spread = spreads[-1]
        zscore = (current_spread - mean_spread) / std_spread

        # 限制在合理范围
        return float(np.clip(zscore, -5.0, 5.0))

    def _kalman_filter(self, prices: np.ndarray) -> tuple:
        """
        执行一维卡尔曼滤波

        Returns:
            filtered_prices: 滤波后的价格估计
            spreads: 每个时刻的偏离度 (观测值 - 估计值)
        """
        n = len(prices)

        # 初始化
        x_hat = prices[0]  # 状态估计
        p = 1.0  # 估计误差协方差

        Q = self.process_noise  # 过程噪声方差
        R = self.measurement_noise  # 观测噪声方差

        filtered_prices = np.zeros(n)
        spreads = np.zeros(n)

        for i in range(n):
            z = prices[i]  # 当前观测值

            # 预测步骤 (Prediction)
            x_hat_minus = x_hat
            p_minus = p + Q

            # 更新步骤 (Update)
            # 卡尔曼增益
            k = p_minus / (p_minus + R)

            # 计算偏离度（在更新前）
            spread = z - x_hat_minus

            # 更新状态估计
            x_hat = x_hat_minus + k * spread

            # 更新估计误差协方差
            p = (1 - k) * p_minus

            filtered_prices[i] = x_hat
            spreads[i] = spread

        return filtered_prices, spreads

    def get_filtered_price(self, data: pd.DataFrame) -> pd.Series:
        """
        获取滤波后的价格序列（用于可视化或调试）

        Args:
            data: 原始数据

        Returns:
            pd.Series: 滤波后的价格序列
        """
        if len(data) < 2:
            return pd.Series(dtype=float)

        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return pd.Series(dtype=float)

        prices = data[close_col].values
        filtered_prices, _ = self._kalman_filter(prices)

        return pd.Series(filtered_prices, index=data.index, name='kalman_filtered')


class KalmanTrendFollower(TimeSeriesFactorBase):
    """
    卡尔曼滤波趋势跟踪因子

    基于卡尔曼滤波器的斜率估计趋势方向和强度。
    使用状态扩展模型，同时估计价格水平和变化率（趋势）。

    状态向量: [price, velocity]
    价格沿趋势方向变化: price(t) = price(t-1) + velocity(t-1) * dt + noise
    """

    def __init__(self, window: int = 60, process_noise: float = 1e-4,
                 measurement_noise: float = 1e-2):
        """
        Args:
            window: 回看窗口长度
            process_noise: 过程噪声方差
            measurement_noise: 观测噪声方差
        """
        super().__init__(name=f"KalmanTrend_{window}", window=window)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def set_params(self, **kwargs) -> 'KalmanTrendFollower':
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def calculate(self, data: pd.DataFrame) -> float:
        """
        计算卡尔曼趋势强度。

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            float: 趋势信号
                   正值: 上升趋势
                   负值: 下降趋势
                   绝对值越大，趋势越强
        """
        if len(data) < self.window:
            return 0.0

        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return 0.0

        prices = data[close_col].values[-self.window:]

        # 运行扩展卡尔曼滤波
        velocity = self._kalman_filter_with_velocity(prices)

        # 归一化速度（相对于价格水平）
        price_level = np.mean(prices)
        if price_level < 1e-10:
            return 0.0

        # 转换为日收益率尺度的信号
        normalized_velocity = velocity / price_level * 100

        # 限制在合理范围
        return float(np.clip(normalized_velocity, -1.0, 1.0))

    def _kalman_filter_with_velocity(self, prices: np.ndarray) -> float:
        """
        二维卡尔曼滤波，同时估计价格和速度

        Returns:
            velocity: 最终估计的价格变化速度
        """
        n = len(prices)

        # 状态向量初始化 [price, velocity]
        x_hat = np.array([prices[0], 0.0])

        # 状态协方差矩阵
        P = np.eye(2) * 1.0

        # 状态转移矩阵 (dt = 1)
        F = np.array([[1, 1],
                      [0, 1]])

        # 观测矩阵 (只观测价格)
        H = np.array([[1, 0]])

        # 过程噪声协方差
        Q = np.array([[self.process_noise, 0],
                      [0, self.process_noise * 0.1]])

        # 观测噪声协方差
        R = np.array([[self.measurement_noise]])

        for i in range(n):
            z = prices[i]

            # 预测
            x_hat_minus = F @ x_hat
            P_minus = F @ P @ F.T + Q

            # 卡尔曼增益
            S = H @ P_minus @ H.T + R
            K = P_minus @ H.T @ np.linalg.inv(S)

            # 更新
            y = z - H @ x_hat_minus  # 残差
            x_hat = x_hat_minus + K.flatten() * y
            P = (np.eye(2) - K @ H) @ P_minus

        return x_hat[1]  # 返回最终的速度估计
