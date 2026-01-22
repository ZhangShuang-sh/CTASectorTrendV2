#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子数学工具模块

提供因子计算中常用的数学函数，包括：
- Copula 模型工具
- 移动平均
- 统计标准化
- 收益率计算
- 相关性计算
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, Callable, List


# =============================================================================
# Copula 模型工具类
# =============================================================================

class CopulaUtils:
    """
    Copula 模型数学工具类

    支持的 Copula 类型:
    - Clayton: 适合下尾相关性
    - Gumbel: 适合上尾相关性
    - Frank: 对称相关性
    """

    @staticmethod
    def clayton_copula(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """
        Clayton Copula 函数

        C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)

        Args:
            u, v: 均匀分布随机变量 (0, 1)
            theta: 参数 θ > 0
        """
        return np.maximum(u ** (-theta) + v ** (-theta) - 1, 1e-10) ** (-1 / theta)

    @staticmethod
    def gumbel_copula(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """
        Gumbel Copula 函数

        C(u,v) = exp(-((-log u)^θ + (-log v)^θ)^(1/θ))

        Args:
            u, v: 均匀分布随机变量 (0, 1)
            theta: 参数 θ >= 1
        """
        return np.exp(-((-np.log(np.maximum(u, 1e-10))) ** theta +
                        (-np.log(np.maximum(v, 1e-10))) ** theta) ** (1 / theta))

    @staticmethod
    def frank_copula(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """
        Frank Copula 函数

        C(u,v) = -1/θ * log(1 + (e^(-θu) - 1)(e^(-θv) - 1) / (e^(-θ) - 1))

        Args:
            u, v: 均匀分布随机变量 (0, 1)
            theta: 参数 θ ≠ 0
        """
        return -1 / theta * np.log(1 + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) /
                                   (np.exp(-theta) - 1))

    @staticmethod
    def clayton_conditional_prob(u: float, v: float, theta: float) -> float:
        """Clayton Copula 条件概率 ∂C/∂u"""
        return v ** (-theta - 1) * (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta - 1)

    @staticmethod
    def gumbel_conditional_prob(u: float, v: float, theta: float) -> float:
        """Gumbel Copula 条件概率 ∂C/∂u"""
        u = max(u, 1e-10)
        v = max(v, 1e-10)
        A = (-np.log(u)) ** theta + (-np.log(v)) ** theta
        return (np.exp(-A ** (1 / theta)) * (-np.log(u)) ** (theta - 1) * A ** (1 / theta - 1)) / u

    @staticmethod
    def frank_conditional_prob(u: float, v: float, theta: float) -> float:
        """Frank Copula 条件概率 ∂C/∂u"""
        numerator = (np.exp(-theta * v) - 1) * np.exp(-theta * u)
        denominator = (np.exp(-theta) - 1) + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        return numerator / max(denominator, 1e-10)

    @staticmethod
    def calculate_ecdf(data: np.ndarray) -> Callable[[float], float]:
        """
        计算经验分布函数 (ECDF)

        Args:
            data: 数据数组

        Returns:
            ecdf: 经验分布函数
        """
        sorted_data = np.sort(data)
        n = len(sorted_data)

        def ecdf(x):
            return np.searchsorted(sorted_data, x, side='right') / n

        return ecdf

    @staticmethod
    def fit_copula(u: np.ndarray, v: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        拟合最优 Copula 函数

        通过最大似然估计选择最优的 Copula 类型和参数。

        Args:
            u, v: 均匀分布随机变量数组

        Returns:
            (copula_type, theta): 最优 Copula 类型和参数
        """
        copulas = {
            'clayton': {
                'func': CopulaUtils.clayton_copula,
                'theta_range': (0.1, 10),
                'theta0': 1.0
            },
            'gumbel': {
                'func': CopulaUtils.gumbel_copula,
                'theta_range': (1.1, 10),
                'theta0': 2.0
            },
            'frank': {
                'func': CopulaUtils.frank_copula,
                'theta_range': (-10, 10),
                'theta0': 1.0
            }
        }

        best_copula = None
        best_theta = None
        best_loglik = -np.inf

        for name, copula_info in copulas.items():
            def neg_loglikelihood(theta):
                c_values = copula_info['func'](u, v, theta)
                c_values = np.clip(c_values, 1e-10, 1 - 1e-10)
                loglik = np.sum(np.log(c_values))
                return -loglik

            try:
                result = minimize(
                    neg_loglikelihood,
                    x0=[copula_info['theta0']],
                    bounds=[copula_info['theta_range']],
                    method='L-BFGS-B'
                )

                if result.success and -result.fun > best_loglik:
                    best_loglik = -result.fun
                    best_copula = name
                    best_theta = result.x[0]
            except Exception:
                continue

        return best_copula, best_theta

    @staticmethod
    def fit_mixed_copula(u: np.ndarray, v: np.ndarray) -> Dict[str, Dict]:
        """
        拟合混合 Copula (线性组合)

        Args:
            u, v: 均匀分布随机变量数组

        Returns:
            Dict: {copula_name: {'theta': float, 'loglik': float}, ...}
        """
        copula_results = {}

        copula_configs = {
            'clayton': (CopulaUtils.clayton_copula, [(0.1, 10)]),
            'gumbel': (CopulaUtils.gumbel_copula, [(1.1, 10)]),
            'frank': (CopulaUtils.frank_copula, [(-10, 10)])
        }

        for name, (func, bounds) in copula_configs.items():
            def neg_loglikelihood(theta):
                try:
                    c_values = func(u, v, theta[0])
                    c_values = np.clip(c_values, 1e-10, 1 - 1e-10)
                    return -np.sum(np.log(c_values))
                except Exception:
                    return np.inf

            try:
                result = minimize(neg_loglikelihood, x0=[1.0], bounds=bounds, method='L-BFGS-B')
                if result.success:
                    copula_results[name] = {
                        'theta': result.x[0],
                        'loglik': -result.fun
                    }
            except Exception:
                continue

        return copula_results

    @staticmethod
    def get_conditional_prob_func(copula_type: str) -> Callable:
        """获取条件概率函数"""
        funcs = {
            'clayton': CopulaUtils.clayton_conditional_prob,
            'gumbel': CopulaUtils.gumbel_conditional_prob,
            'frank': CopulaUtils.frank_conditional_prob
        }
        return funcs.get(copula_type, CopulaUtils.frank_conditional_prob)


# =============================================================================
# 通用数学工具函数
# =============================================================================

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    计算简单移动平均

    Args:
        data: 输入数据数组
        window: 窗口大小

    Returns:
        移动平均数组 (长度 = len(data) - window + 1)
    """
    if len(data) < window:
        return np.array([])
    return np.convolve(data, np.ones(window) / window, mode='valid')


def exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """
    计算指数移动平均

    Args:
        data: 输入数据数组
        span: EMA 跨度

    Returns:
        EMA 数组
    """
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_zscore(data: pd.Series, window: int = None) -> pd.Series:
    """
    计算 Z-Score 标准化

    Args:
        data: 输入数据
        window: 滚动窗口 (None 表示全样本)

    Returns:
        Z-Score 标准化后的数据
    """
    if window is None:
        mean_val = data.mean()
        std_val = data.std()
    else:
        mean_val = data.rolling(window).mean()
        std_val = data.rolling(window).std()

    std_val = std_val.replace(0, np.nan)
    return (data - mean_val) / std_val


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    计算收益率

    Args:
        prices: 价格序列
        method: 'simple' (简单收益率) 或 'log' (对数收益率)

    Returns:
        收益率序列
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def calculate_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    计算滚动波动率

    Args:
        returns: 收益率序列
        window: 滚动窗口
        annualize: 是否年化

    Returns:
        波动率序列
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def calculate_correlation_matrix(
    data_dict: Dict[str, np.ndarray],
    min_periods: int = 20
) -> Dict[Tuple[str, str], float]:
    """
    计算资产间相关性矩阵

    Args:
        data_dict: {ticker: returns_array}
        min_periods: 最小数据长度要求

    Returns:
        {(ticker1, ticker2): correlation}
    """
    correlations = {}
    tickers = list(data_dict.keys())

    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i + 1:]:
            r1 = data_dict[ticker1]
            r2 = data_dict[ticker2]

            # 对齐长度
            min_len = min(len(r1), len(r2))
            if min_len < min_periods:
                continue

            r1 = r1[-min_len:]
            r2 = r2[-min_len:]

            corr = np.corrcoef(r1, r2)[0, 1]
            if not np.isnan(corr):
                correlations[(ticker1, ticker2)] = corr
                correlations[(ticker2, ticker1)] = corr

    return correlations


def rank_normalize(data: pd.Series) -> pd.Series:
    """
    Rank 归一化到 [0, 1]

    Args:
        data: 输入数据

    Returns:
        归一化后的数据
    """
    return data.rank(pct=True)


def winsorize(data: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize 处理极端值

    Args:
        data: 输入数据
        lower: 下分位数
        upper: 上分位数

    Returns:
        处理后的数据
    """
    lower_val = data.quantile(lower)
    upper_val = data.quantile(upper)
    return data.clip(lower=lower_val, upper=upper_val)


# =============================================================================
# Volatility Calculator for Risk Parity / Volatility Parity
# =============================================================================

class VolatilityCalculator:
    """
    波动率计算器 - 用于风险平价/波动率平价仓位计算

    支持两种波动率测量方法:
    - Standard Deviation (StdDev): 标准差
    - ATR (Average True Range): 平均真实波幅

    使用场景:
    - Pair Trading: 波动率平衡仓位 (Position_A * Price_A * Vol_A = Position_B * Price_B * Vol_B)
    - Multi-Asset: 逆波动率加权 (weight = 1 / volatility)
    """

    def __init__(
        self,
        window: int = 20,
        method: str = 'stddev',
        annualize: bool = True,
        min_periods: int = 10,
        default_volatility: float = 0.15
    ):
        """
        Args:
            window: 波动率计算窗口 (默认20天)
            method: 'stddev' (标准差) 或 'atr' (平均真实波幅)
            annualize: 是否年化 (默认True)
            min_periods: 最小数据要求
            default_volatility: 数据不足时的默认波动率
        """
        self.window = window
        self.method = method
        self.annualize = annualize
        self.min_periods = min_periods
        self.default_volatility = default_volatility

    def calculate_asset_volatility(
        self,
        prices: pd.Series,
        high: pd.Series = None,
        low: pd.Series = None
    ) -> float:
        """
        计算单个资产的波动率

        Args:
            prices: 收盘价序列
            high: 最高价序列 (ATR方法需要)
            low: 最低价序列 (ATR方法需要)

        Returns:
            年化波动率 (或非年化, 取决于annualize参数)
        """
        if prices is None or len(prices) < self.min_periods:
            return self.default_volatility

        try:
            if self.method == 'atr' and high is not None and low is not None:
                vol = self._calculate_atr(prices, high, low)
            else:
                vol = self._calculate_stddev(prices)

            if pd.isna(vol) or vol <= 0:
                return self.default_volatility

            return vol

        except Exception:
            return self.default_volatility

    def _calculate_stddev(self, prices: pd.Series) -> float:
        """使用标准差计算波动率"""
        returns = prices.pct_change().dropna()

        if len(returns) < self.min_periods:
            return self.default_volatility

        # 使用最近window天的数据
        recent_returns = returns.tail(self.window)
        daily_vol = recent_returns.std()

        if self.annualize:
            return daily_vol * np.sqrt(252)
        return daily_vol

    def _calculate_atr(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series
    ) -> float:
        """
        使用ATR计算波动率

        ATR = Average(max(high-low, |high-prev_close|, |low-prev_close|))
        """
        if len(close) < self.min_periods:
            return self.default_volatility

        prev_close = close.shift(1)

        tr1 = high - low  # 当日波幅
        tr2 = (high - prev_close).abs()  # 上涨缺口
        tr3 = (low - prev_close).abs()  # 下跌缺口

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = 平均真实波幅
        atr = true_range.rolling(self.window).mean().dropna()

        if len(atr) == 0:
            return self.default_volatility

        # ATR 作为价格百分比
        atr_pct = atr.iloc[-1] / close.iloc[-1]

        if self.annualize:
            return atr_pct * np.sqrt(252)
        return atr_pct

    def calculate_pair_volatility_weights(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        high1: pd.Series = None,
        low1: pd.Series = None,
        high2: pd.Series = None,
        low2: pd.Series = None
    ) -> Tuple[float, float]:
        """
        计算配对交易的波动率平衡权重

        实现: Position_A * Price_A * Vol_A = Position_B * Price_B * Vol_B

        返回归一化权重, 使得:
        weight1 * price1 * vol1 = weight2 * price2 * vol2

        Args:
            prices1, prices2: 两个资产的价格序列
            high1, low1, high2, low2: 最高/最低价 (ATR方法)

        Returns:
            (weight1, weight2): 归一化权重, 和为1
        """
        vol1 = self.calculate_asset_volatility(prices1, high1, low1)
        vol2 = self.calculate_asset_volatility(prices2, high2, low2)

        if vol1 <= 0 or vol2 <= 0:
            return 0.5, 0.5

        # 逆波动率权重
        inv_vol1 = 1.0 / vol1
        inv_vol2 = 1.0 / vol2

        total_inv_vol = inv_vol1 + inv_vol2

        weight1 = inv_vol1 / total_inv_vol
        weight2 = inv_vol2 / total_inv_vol

        return weight1, weight2

    def calculate_pair_position_sizes(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        allocated_capital: float,
        margin_rate: float = 0.1,
        high1: pd.Series = None,
        low1: pd.Series = None,
        high2: pd.Series = None,
        low2: pd.Series = None
    ) -> Tuple[int, int]:
        """
        计算波动率平衡的配对仓位数量

        公式: Qty1 * Price1 * Vol1 = Qty2 * Price2 * Vol2

        Args:
            prices1, prices2: 两个资产的价格序列 (最新价格用于计算)
            allocated_capital: 分配给这对的总资金
            margin_rate: 保证金率
            high1, low1, high2, low2: 最高/最低价 (ATR方法)

        Returns:
            (quantity1, quantity2): 波动率平衡后的仓位数量
        """
        if prices1 is None or prices2 is None:
            return 1, 1

        price1 = prices1.iloc[-1] if isinstance(prices1, pd.Series) else float(prices1)
        price2 = prices2.iloc[-1] if isinstance(prices2, pd.Series) else float(prices2)

        if price1 <= 0 or price2 <= 0:
            return 1, 1

        # 获取波动率权重
        weight1, weight2 = self.calculate_pair_volatility_weights(
            prices1, prices2, high1, low1, high2, low2
        )

        # 分配资金到每腿
        capital1 = allocated_capital * weight1
        capital2 = allocated_capital * weight2

        # 计算数量 (考虑保证金)
        quantity1 = int(capital1 / (price1 * margin_rate))
        quantity2 = int(capital2 / (price2 * margin_rate))

        # 确保最小交易单位
        quantity1 = max(1, quantity1)
        quantity2 = max(1, quantity2)

        return quantity1, quantity2

    def calculate_multi_asset_weights(
        self,
        volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算多资产的逆波动率权重 (Risk Parity)

        公式: weight_i = (1 / vol_i) / sum(1 / vol_j)

        Args:
            volatilities: {asset: volatility}

        Returns:
            {asset: weight}: 归一化权重, 和为1
        """
        if not volatilities:
            return {}

        # 过滤无效波动率
        valid_vols = {
            asset: vol for asset, vol in volatilities.items()
            if vol is not None and vol > 0 and not pd.isna(vol)
        }

        if not valid_vols:
            # 所有资产等权
            n = len(volatilities)
            return {asset: 1.0 / n for asset in volatilities}

        # 逆波动率权重
        inv_vols = {asset: 1.0 / vol for asset, vol in valid_vols.items()}
        total_inv_vol = sum(inv_vols.values())

        weights = {}
        for asset in volatilities:
            if asset in inv_vols:
                weights[asset] = inv_vols[asset] / total_inv_vol
            else:
                # 无效波动率的资产使用默认波动率
                weights[asset] = (1.0 / self.default_volatility) / (total_inv_vol + 1.0 / self.default_volatility)

        # 重新归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def calculate_volatility_from_dataframe(
        self,
        df: pd.DataFrame,
        date_col: str = 'TRADE_DT',
        price_col: str = 'S_DQ_CLOSE',
        high_col: str = 'S_DQ_HIGH',
        low_col: str = 'S_DQ_LOW',
        current_date: pd.Timestamp = None
    ) -> float:
        """
        从DataFrame计算波动率 (适配项目数据格式)

        Args:
            df: 包含价格数据的DataFrame
            date_col: 日期列名
            price_col: 收盘价列名
            high_col: 最高价列名
            low_col: 最低价列名
            current_date: 当前日期 (过滤用)

        Returns:
            波动率
        """
        if df is None or df.empty:
            return self.default_volatility

        try:
            # 过滤日期
            if current_date is not None and date_col in df.columns:
                df = df[df[date_col] <= current_date]

            if len(df) < self.min_periods:
                return self.default_volatility

            # 按日期排序
            df = df.sort_values(date_col).tail(self.window * 2)

            prices = df[price_col] if price_col in df.columns else None
            high = df[high_col] if high_col in df.columns else None
            low = df[low_col] if low_col in df.columns else None

            return self.calculate_asset_volatility(prices, high, low)

        except Exception:
            return self.default_volatility


def create_volatility_calculator(
    window: int = 20,
    method: str = 'stddev',
    annualize: bool = True
) -> VolatilityCalculator:
    """
    工厂函数: 创建波动率计算器

    Args:
        window: 波动率窗口
        method: 'stddev' 或 'atr'
        annualize: 是否年化

    Returns:
        VolatilityCalculator 实例
    """
    return VolatilityCalculator(
        window=window,
        method=method,
        annualize=annualize
    )
