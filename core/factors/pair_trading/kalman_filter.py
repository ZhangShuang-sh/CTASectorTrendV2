#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡尔曼滤波配对交易因子

基于卡尔曼滤波的配对交易策略，利用动态对冲比率和价差均值回归生成交易信号。

参考: 中信期货金融工程专题报告《配对交易专题（二）：卡尔曼滤波在价差套利中的应用》20240129
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from core.factors.pair_trading.base import PairTradingFactorBase


@dataclass
class KalmanFilterState:
    """卡尔曼滤波器状态"""
    beta: float = 0.0           # 对冲比率 (斜率)
    intercept: float = 0.0      # 截距
    P_beta: float = 1.0         # beta协方差
    P_intercept: float = 1.0    # intercept协方差
    spread_error: float = 0.0   # 价差预测误差


class KalmanFilter:
    """
    卡尔曼滤波器 - 用于动态估计对冲比率

    状态空间模型:
    - 观测方程: y_t = alpha + beta * x_t + epsilon_t
    - 状态方程: [alpha_t, beta_t]' = [alpha_{t-1}, beta_{t-1}]' + omega_t

    其中 y_t 是资产1价格, x_t 是资产2价格
    """

    def __init__(
        self,
        delta: float = 0.0001,
        Ve: float = 0.001,
        estimate_intercept: bool = True
    ):
        """
        Args:
            delta: 状态转移噪声方差 (控制beta变化速度)
            Ve: 观测噪声方差
            estimate_intercept: 是否估计截距
        """
        self.delta = delta
        self.Ve = Ve
        self.estimate_intercept = estimate_intercept

        # 状态变量
        self.beta = 0.0
        self.intercept = 0.0
        self.P_beta = 1.0
        self.P_intercept = 1.0

        # 历史记录
        self.beta_history: List[float] = []
        self.spread_history: List[float] = []
        self.error_history: List[float] = []

    def reset(self):
        """重置滤波器状态"""
        self.beta = 0.0
        self.intercept = 0.0
        self.P_beta = 1.0
        self.P_intercept = 1.0
        self.beta_history = []
        self.spread_history = []
        self.error_history = []

    def update(self, y: float, x: float) -> Tuple[float, float, float]:
        """
        卡尔曼滤波单步更新

        Args:
            y: 资产1价格 (被解释变量)
            x: 资产2价格 (解释变量)

        Returns:
            (beta, spread, error): 更新后的对冲比率、价差、预测误差
        """
        # 预测步骤 (状态转移)
        Q_beta = self.delta
        Q_intercept = self.delta if self.estimate_intercept else 0

        P_beta_pred = self.P_beta + Q_beta
        P_intercept_pred = self.P_intercept + Q_intercept

        # 预测观测值
        if self.estimate_intercept:
            y_pred = self.intercept + self.beta * x
        else:
            y_pred = self.beta * x

        # 预测误差 (创新)
        error = y - y_pred

        # 预测误差方差
        S = x**2 * P_beta_pred + P_intercept_pred + self.Ve

        # 卡尔曼增益
        K_beta = P_beta_pred * x / S
        K_intercept = P_intercept_pred / S if self.estimate_intercept else 0

        # 更新状态
        self.beta = self.beta + K_beta * error
        if self.estimate_intercept:
            self.intercept = self.intercept + K_intercept * error

        # 更新协方差
        self.P_beta = (1 - K_beta * x) * P_beta_pred
        if self.estimate_intercept:
            self.P_intercept = (1 - K_intercept) * P_intercept_pred

        # 计算价差
        spread = y - self.beta * x - (self.intercept if self.estimate_intercept else 0)

        # 记录历史
        self.beta_history.append(self.beta)
        self.spread_history.append(spread)
        self.error_history.append(error)

        return self.beta, spread, error

    def fit(self, y_series: np.ndarray, x_series: np.ndarray) -> None:
        """
        对整个序列进行卡尔曼滤波

        Args:
            y_series: 资产1价格序列
            x_series: 资产2价格序列
        """
        self.reset()

        for y, x in zip(y_series, x_series):
            self.update(y, x)

    def get_state(self) -> KalmanFilterState:
        """获取当前状态"""
        return KalmanFilterState(
            beta=self.beta,
            intercept=self.intercept,
            P_beta=self.P_beta,
            P_intercept=self.P_intercept,
            spread_error=self.error_history[-1] if self.error_history else 0.0
        )


class KalmanFilterPairFactor(PairTradingFactorBase):
    """
    卡尔曼滤波配对交易因子

    使用卡尔曼滤波动态估计对冲比率，基于价差的z-score生成交易信号。

    支持三种进出场规则:
    - Rule 1: 经典规则 - zscore穿越固定阈值
    - Rule 2: 增强规则 - zscore穿越阈值 + 价格分位条件 + 止盈止损
    - Rule 3: 混合规则 - Rule1建仓 + Rule2平仓

    信号含义:
    - +1: 做多价差 (Long asset1, Short asset2)
    - -1: 做空价差 (Short asset1, Long asset2)
    -  0: 无信号 / 平仓
    """

    def __init__(
        self,
        name: str = "KalmanFilterPairFactor",
        window: int = 60,
        zscore_window: int = 20,
        entry_threshold: float = 1.0,
        exit_threshold: float = 0.0,
        rule: int = 1,
        # 卡尔曼滤波参数
        delta: float = 0.0001,
        Ve: float = 0.001,
        estimate_intercept: bool = True,
        # Rule 2/3 参数
        quantile_high: float = 0.75,
        quantile_low: float = 0.25,
        stop_profit: float = 0.05,
        stop_loss: float = 0.05,
        max_holding_period: int = 45,
    ):
        """
        Args:
            name: 因子名称
            window: 数据回看窗口
            zscore_window: zscore计算窗口
            entry_threshold: 开仓阈值 (zscore绝对值)
            exit_threshold: 平仓阈值 (Rule1) 或 无效 (Rule2/3)
            rule: 进出场规则 (1, 2, 3)
            delta: 卡尔曼滤波状态转移噪声
            Ve: 卡尔曼滤波观测噪声
            estimate_intercept: 是否估计截距
            quantile_high: 高分位阈值 (Rule 2)
            quantile_low: 低分位阈值 (Rule 2)
            stop_profit: 止盈比例 (Rule 2/3)
            stop_loss: 止损比例 (Rule 2/3)
            max_holding_period: 最大持有期 (Rule 2/3)
        """
        super().__init__(
            name=name,
            window=window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold
        )

        self.zscore_window = zscore_window
        self.rule = rule

        # 卡尔曼滤波参数
        self.delta = delta
        self.Ve = Ve
        self.estimate_intercept = estimate_intercept

        # Rule 2/3 参数
        self.quantile_high = quantile_high
        self.quantile_low = quantile_low
        self.stop_profit = stop_profit
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period

        # 卡尔曼滤波器实例
        self._kalman_filter = KalmanFilter(
            delta=delta,
            Ve=Ve,
            estimate_intercept=estimate_intercept
        )

        # 持仓状态 (用于Rule 2/3)
        self._position = 0  # +1, -1, 0
        self._entry_price_spread = 0.0
        self._holding_days = 0

        # 更新参数字典
        self._params.update({
            'zscore_window': zscore_window,
            'rule': rule,
            'delta': delta,
            'Ve': Ve,
            'estimate_intercept': estimate_intercept,
            'quantile_high': quantile_high,
            'quantile_low': quantile_low,
            'stop_profit': stop_profit,
            'stop_loss': stop_loss,
            'max_holding_period': max_holding_period,
        })

    def set_params(self, **kwargs) -> 'KalmanFilterPairFactor':
        """动态设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 更新卡尔曼滤波器参数
        if 'delta' in kwargs or 'Ve' in kwargs or 'estimate_intercept' in kwargs:
            self._kalman_filter = KalmanFilter(
                delta=self.delta,
                Ve=self.Ve,
                estimate_intercept=self.estimate_intercept
            )

        return self

    def reset_position_state(self):
        """重置持仓状态"""
        self._position = 0
        self._entry_price_spread = 0.0
        self._holding_days = 0

    def calculate(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        date: pd.Timestamp = None
    ) -> float:
        """
        计算配对信号

        Args:
            df1: 第一个资产的历史数据 (被解释变量)
            df2: 第二个资产的历史数据 (解释变量)
            date: 当前日期

        Returns:
            float: 信号强度值 [-1.0, 1.0]
        """
        # 提取价格数据
        prices1 = self._extract_prices(df1, date)
        prices2 = self._extract_prices(df2, date)

        if prices1 is None or prices2 is None:
            return 0.0

        min_len = min(len(prices1), len(prices2))
        if min_len < self.window:
            return 0.0

        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]

        # 运行卡尔曼滤波
        self._kalman_filter.reset()
        self._kalman_filter.fit(prices1, prices2)

        # 获取价差序列和当前状态
        spread_series = np.array(self._kalman_filter.spread_history)
        error_series = np.array(self._kalman_filter.error_history)

        if len(spread_series) < self.zscore_window:
            return 0.0

        # 计算zscore
        zscore = self._compute_zscore(error_series, self.zscore_window)

        if np.isnan(zscore):
            return 0.0

        # 根据规则生成信号
        if self.rule == 1:
            signal = self._generate_signal_rule1(zscore)
        elif self.rule == 2:
            signal = self._generate_signal_rule2(
                zscore, prices1, prices2, spread_series[-1]
            )
        elif self.rule == 3:
            signal = self._generate_signal_rule3(
                zscore, spread_series[-1]
            )
        else:
            signal = self._generate_signal_rule1(zscore)

        return float(np.clip(signal, -1.0, 1.0))

    def _extract_prices(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp
    ) -> Optional[np.ndarray]:
        """提取价格序列"""
        if df is None or len(df) < self.window:
            return None

        # 筛选截止到date的数据
        if date is not None and isinstance(df.index, pd.DatetimeIndex):
            df_until = df[df.index <= date]
        else:
            df_until = df

        if len(df_until) < self.window:
            return None

        # 获取收盘价
        if 'close' in df_until.columns:
            prices = df_until['close'].values
        elif 'S_DQ_CLOSE' in df_until.columns:
            prices = df_until['S_DQ_CLOSE'].values
        else:
            return None

        return prices[-self.window:]

    def _compute_zscore(self, series: np.ndarray, window: int) -> float:
        """计算zscore"""
        if len(series) < window:
            return np.nan

        recent = series[-window:]
        mean_val = np.mean(recent)
        std_val = np.std(recent)

        if std_val < 1e-10:
            return 0.0

        zscore = (series[-1] - mean_val) / std_val
        return zscore

    def _generate_signal_rule1(self, zscore: float) -> float:
        """
        规则1: 经典进出场

        - zscore < -threshold: 做多价差 (+1)
        - zscore > +threshold: 做空价差 (-1)
        - otherwise: 无信号 (0)
        """
        if zscore < -self.entry_threshold:
            return 1.0  # 做多价差
        elif zscore > self.entry_threshold:
            return -1.0  # 做空价差
        elif abs(zscore) < self.exit_threshold:
            return 0.0  # 平仓
        else:
            return 0.0  # 持仓不变 (简化处理)

    def _generate_signal_rule2(
        self,
        zscore: float,
        prices1: np.ndarray,
        prices2: np.ndarray,
        current_spread: float
    ) -> float:
        """
        规则2: 增强进出场

        建仓:
        - zscore < -threshold AND 两价格 < 各自历史高分位: 做多价差
        - zscore > +threshold AND 两价格 > 各自历史低分位: 做空价差

        平仓:
        - 止盈止损 OR 超过最大持有期
        """
        # 计算分位数
        quantile_high_1 = np.percentile(prices1, self.quantile_high * 100)
        quantile_low_1 = np.percentile(prices1, self.quantile_low * 100)
        quantile_high_2 = np.percentile(prices2, self.quantile_high * 100)
        quantile_low_2 = np.percentile(prices2, self.quantile_low * 100)

        current_price1 = prices1[-1]
        current_price2 = prices2[-1]

        # 更新持仓天数
        if self._position != 0:
            self._holding_days += 1

        # 检查平仓条件 (Rule 2)
        if self._position != 0:
            # 计算盈亏
            pnl_ratio = (current_spread - self._entry_price_spread) / abs(self._entry_price_spread + 1e-10)
            if self._position == -1:
                pnl_ratio = -pnl_ratio

            # 止盈止损
            if pnl_ratio >= self.stop_profit or pnl_ratio <= -self.stop_loss:
                self._position = 0
                self._entry_price_spread = 0.0
                self._holding_days = 0
                return 0.0

            # 超过最大持有期
            if self._holding_days >= self.max_holding_period:
                self._position = 0
                self._entry_price_spread = 0.0
                self._holding_days = 0
                return 0.0

            # 继续持仓
            return float(self._position)

        # 建仓条件
        if zscore < -self.entry_threshold:
            # 做多价差条件: 两价格尚未达到历史高分位
            if current_price1 < quantile_high_1 and current_price2 < quantile_high_2:
                self._position = 1
                self._entry_price_spread = current_spread
                self._holding_days = 0
                return 1.0

        elif zscore > self.entry_threshold:
            # 做空价差条件: 两价格已脱离历史低分位
            if current_price1 > quantile_low_1 and current_price2 > quantile_low_2:
                self._position = -1
                self._entry_price_spread = current_spread
                self._holding_days = 0
                return -1.0

        return 0.0

    def _generate_signal_rule3(
        self,
        zscore: float,
        current_spread: float
    ) -> float:
        """
        规则3: 混合进出场

        建仓: Rule 1逻辑
        平仓: Rule 2逻辑 (止盈止损 + 最大持有期)
        """
        # 更新持仓天数
        if self._position != 0:
            self._holding_days += 1

        # 检查平仓条件 (同Rule 2)
        if self._position != 0:
            pnl_ratio = (current_spread - self._entry_price_spread) / abs(self._entry_price_spread + 1e-10)
            if self._position == -1:
                pnl_ratio = -pnl_ratio

            if pnl_ratio >= self.stop_profit or pnl_ratio <= -self.stop_loss:
                self._position = 0
                self._entry_price_spread = 0.0
                self._holding_days = 0
                return 0.0

            if self._holding_days >= self.max_holding_period:
                self._position = 0
                self._entry_price_spread = 0.0
                self._holding_days = 0
                return 0.0

            return float(self._position)

        # 建仓条件 (同Rule 1)
        if zscore < -self.entry_threshold:
            self._position = 1
            self._entry_price_spread = current_spread
            self._holding_days = 0
            return 1.0
        elif zscore > self.entry_threshold:
            self._position = -1
            self._entry_price_spread = current_spread
            self._holding_days = 0
            return -1.0

        return 0.0

    def get_kalman_state(self) -> KalmanFilterState:
        """获取当前卡尔曼滤波器状态"""
        return self._kalman_filter.get_state()

    def get_beta_history(self) -> List[float]:
        """获取对冲比率历史"""
        return self._kalman_filter.beta_history.copy()

    def get_spread_history(self) -> List[float]:
        """获取价差历史"""
        return self._kalman_filter.spread_history.copy()

    def compute_hedge_ratio(
        self,
        prices1: np.ndarray,
        prices2: np.ndarray
    ) -> float:
        """
        计算当前对冲比率

        Args:
            prices1: 资产1价格序列
            prices2: 资产2价格序列

        Returns:
            对冲比率 beta
        """
        self._kalman_filter.reset()
        self._kalman_filter.fit(prices1, prices2)
        return self._kalman_filter.beta


def create_kalman_filter_factor(
    rule: int = 1,
    entry_threshold: float = 1.0,
    zscore_window: int = 20,
    **kwargs
) -> KalmanFilterPairFactor:
    """
    工厂函数: 创建卡尔曼滤波配对因子

    Args:
        rule: 规则类型 (1, 2, 3)
        entry_threshold: 开仓阈值
        zscore_window: zscore窗口
        **kwargs: 其他参数

    Returns:
        KalmanFilterPairFactor 实例
    """
    # 根据规则设置默认参数
    default_params = {
        1: {
            'entry_threshold': 0.6,
            'zscore_window': 10,
            'exit_threshold': 0.0,
        },
        2: {
            'entry_threshold': 1.2,
            'zscore_window': 90,
            'quantile_high': 0.75,
            'quantile_low': 0.25,
            'stop_profit': 0.05,
            'stop_loss': 0.05,
            'max_holding_period': 45,
        },
        3: {
            'entry_threshold': 0.1,
            'zscore_window': 30,
            'stop_profit': 0.01,
            'stop_loss': 0.01,
            'max_holding_period': 80,
        }
    }

    params = default_params.get(rule, {})
    params.update(kwargs)
    params['rule'] = rule

    if 'entry_threshold' not in kwargs:
        params['entry_threshold'] = params.get('entry_threshold', entry_threshold)
    if 'zscore_window' not in kwargs:
        params['zscore_window'] = params.get('zscore_window', zscore_window)

    return KalmanFilterPairFactor(**params)
