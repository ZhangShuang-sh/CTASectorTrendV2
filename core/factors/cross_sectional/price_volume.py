#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Price Volume Cross-Sectional Factors

CTA量价类截面因子实现。

基于中信期货研究报告《CTA风格因子手册（一）：量价类因子》实现的6个因子:
- VolatilityFactor: 波动率因子
- CVFactor: 收益率变异系数因子
- SkewnessFactor: 偏度因子
- KurtosisFactor: 峰度因子
- AmplitudeFactor: 振幅因子
- LiquidityFactor: 流动性因子

因子方向说明:
- 正向因子 (POSITIVE): 因子值越大，预期收益越高，做多高值做空低值
- 负向因子 (NEGATIVE): 因子值越小，预期收益越高，做空高值做多低值

Source: V1 factors/cross_sectional/price_volume.py (100% logic preserved)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from enum import Enum

from core.factors.cross_sectional.base import CrossSectionalFactor
from core.factors.registry import register


class FactorDirection(Enum):
    """因子方向枚举"""
    POSITIVE = 1   # 正向因子：做多高值，做空低值
    NEGATIVE = -1  # 负向因子：做空高值，做多低值


@register('VolatilityFactor')
class VolatilityFactor(CrossSectionalFactor):
    """
    波动率因子 (Volatility)。

    公式: V_t = sqrt(1/J * sum((R_i - R_mean)^2))
    即过去J日收益率的标准差。

    方向: 正向因子 - 高波动率预期收益更高
    最优参数: J=10，年化收益3.79%，夏普0.40
    """

    direction = FactorDirection.POSITIVE

    def __init__(
        self,
        name: str = "Volatility",
        window: int = 10,
        annualize: bool = False
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认10日)
            annualize: 是否年化 (默认False)
        """
        super().__init__(name=name, window=window)
        self.annualize = annualize
        self._params.update({
            'annualize': annualize,
            'direction': 'POSITIVE'
        })

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """
        计算全市场的波动率因子值。

        Args:
            universe_data: 全市场数据字典
            date: 当前回测截面的日期

        Returns:
            pd.Series: 波动率因子值，已按方向调整排名
        """
        factor_values = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < self.window + 1:
                continue

            df = df.sort_index()
            if date is not None:
                df_until_date = df[df.index <= date]
            else:
                df_until_date = df

            if len(df_until_date) < self.window + 1:
                continue

            # 计算收益率
            if 'returns' in df_until_date.columns:
                returns = df_until_date['returns'].iloc[-self.window:]
            else:
                close = df_until_date['close'] if 'close' in df_until_date.columns else df_until_date['S_DQ_CLOSE']
                returns = close.pct_change().iloc[-self.window:]

            returns = returns.dropna()
            if len(returns) < self.window * 0.8:  # 至少80%数据
                continue

            # 计算标准差
            volatility = returns.std()

            if self.annualize:
                volatility = volatility * np.sqrt(252)

            if not np.isnan(volatility) and not np.isinf(volatility):
                factor_values[ticker] = volatility

        if not factor_values:
            return pd.Series(dtype=float)

        factor_series = pd.Series(factor_values)

        # 正向因子: 高值排名高
        ranks = factor_series.rank(pct=True)

        return ranks

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


@register('CVFactor')
class CVFactor(CrossSectionalFactor):
    """
    收益率变异系数因子 (Coefficient of Variation)。

    公式: CV_t = Var(R) / |R_mean|

    方向: 正向因子 - 高变异系数预期收益更高
    最优参数: J=5，年化收益7.77%，夏普1.08 (表现最优)
    """

    direction = FactorDirection.POSITIVE

    def __init__(
        self,
        name: str = "CV",
        window: int = 5
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认5日)
        """
        super().__init__(name=name, window=window)
        self._params.update({'direction': 'POSITIVE'})

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """计算全市场的CV因子值。"""
        factor_values = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < self.window + 1:
                continue

            df = df.sort_index()
            if date is not None:
                df_until_date = df[df.index <= date]
            else:
                df_until_date = df

            if len(df_until_date) < self.window + 1:
                continue

            # 计算收益率
            if 'returns' in df_until_date.columns:
                returns = df_until_date['returns'].iloc[-self.window:]
            else:
                close = df_until_date['close'] if 'close' in df_until_date.columns else df_until_date['S_DQ_CLOSE']
                returns = close.pct_change().iloc[-self.window:]

            returns = returns.dropna()
            if len(returns) < max(2, self.window * 0.8):
                continue

            # 计算变异系数 = Var / |Mean|
            variance = returns.var()
            mean = returns.mean()

            if mean == 0 or np.isnan(mean):
                continue

            cv = variance / abs(mean)

            if not np.isnan(cv) and not np.isinf(cv):
                factor_values[ticker] = cv

        if not factor_values:
            return pd.Series(dtype=float)

        factor_series = pd.Series(factor_values)
        ranks = factor_series.rank(pct=True)

        return ranks

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


@register('SkewnessFactor')
class SkewnessFactor(CrossSectionalFactor):
    """
    偏度因子 (Skewness)。

    公式: S_t = 1/J * sum(((R_i - R_mean) / sigma)^3)

    方向: 负向因子 - 低偏度预期收益更高
    最优参数: J=10，年化收益3.60%，夏普0.54
    """

    direction = FactorDirection.NEGATIVE

    def __init__(
        self,
        name: str = "Skewness",
        window: int = 10
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认10日)
        """
        super().__init__(name=name, window=window)
        self._params.update({'direction': 'NEGATIVE'})

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """计算全市场的偏度因子值。"""
        factor_values = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < self.window + 1:
                continue

            df = df.sort_index()
            if date is not None:
                df_until_date = df[df.index <= date]
            else:
                df_until_date = df

            if len(df_until_date) < self.window + 1:
                continue

            # 计算收益率
            if 'returns' in df_until_date.columns:
                returns = df_until_date['returns'].iloc[-self.window:]
            else:
                close = df_until_date['close'] if 'close' in df_until_date.columns else df_until_date['S_DQ_CLOSE']
                returns = close.pct_change().iloc[-self.window:]

            returns = returns.dropna()
            if len(returns) < max(3, self.window * 0.8):
                continue

            # 使用pandas内置偏度计算
            skewness = returns.skew()

            if not np.isnan(skewness) and not np.isinf(skewness):
                factor_values[ticker] = skewness

        if not factor_values:
            return pd.Series(dtype=float)

        factor_series = pd.Series(factor_values)

        # 负向因子: 低值排名高，所以用 1 - rank
        ranks = 1 - factor_series.rank(pct=True)

        return ranks

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


@register('KurtosisFactor')
class KurtosisFactor(CrossSectionalFactor):
    """
    峰度因子 (Kurtosis)。

    公式: K_t = 1/J * sum(((R_i - R_mean) / sigma)^4) - 3
    即超额峰度 (excess kurtosis)。

    方向: 负向因子 - 低峰度预期收益更高
    最优参数: J=60，年化收益6.03%，夏普0.90
    """

    direction = FactorDirection.NEGATIVE

    def __init__(
        self,
        name: str = "Kurtosis",
        window: int = 60
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认60日)
        """
        super().__init__(name=name, window=window)
        self._params.update({'direction': 'NEGATIVE'})

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """计算全市场的峰度因子值。"""
        factor_values = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < self.window + 1:
                continue

            df = df.sort_index()
            if date is not None:
                df_until_date = df[df.index <= date]
            else:
                df_until_date = df

            if len(df_until_date) < self.window + 1:
                continue

            # 计算收益率
            if 'returns' in df_until_date.columns:
                returns = df_until_date['returns'].iloc[-self.window:]
            else:
                close = df_until_date['close'] if 'close' in df_until_date.columns else df_until_date['S_DQ_CLOSE']
                returns = close.pct_change().iloc[-self.window:]

            returns = returns.dropna()
            if len(returns) < max(4, self.window * 0.8):
                continue

            # 使用pandas内置峰度计算 (已经是超额峰度，减去了3)
            kurtosis = returns.kurt()

            if not np.isnan(kurtosis) and not np.isinf(kurtosis):
                factor_values[ticker] = kurtosis

        if not factor_values:
            return pd.Series(dtype=float)

        factor_series = pd.Series(factor_values)

        # 负向因子: 低值排名高
        ranks = 1 - factor_series.rank(pct=True)

        return ranks

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


@register('AmplitudeFactor')
class AmplitudeFactor(CrossSectionalFactor):
    """
    振幅因子 (Amplitude)。

    公式: A_t = 1/J * sum((H_i - P_i) / P_i)
    其中 H_i 为最高价，P_i 为收盘价。

    方向: 正向因子 - 高振幅预期收益更高
    最优参数: J=5，年化收益4.62%，夏普0.45
    """

    direction = FactorDirection.POSITIVE

    def __init__(
        self,
        name: str = "Amplitude",
        window: int = 5
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认5日)
        """
        super().__init__(name=name, window=window)
        self._params.update({'direction': 'POSITIVE'})

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """计算全市场的振幅因子值。"""
        factor_values = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < self.window:
                continue

            df = df.sort_index()
            if date is not None:
                df_until_date = df[df.index <= date]
            else:
                df_until_date = df

            if len(df_until_date) < self.window:
                continue

            # 获取最高价和收盘价
            high_col = 'high' if 'high' in df_until_date.columns else 'S_DQ_HIGH'
            close_col = 'close' if 'close' in df_until_date.columns else 'S_DQ_CLOSE'

            if high_col not in df_until_date.columns or close_col not in df_until_date.columns:
                continue

            high = df_until_date[high_col].iloc[-self.window:]
            close = df_until_date[close_col].iloc[-self.window:]

            # 计算振幅 = (High - Close) / Close
            amplitude = ((high - close) / close).replace([np.inf, -np.inf], np.nan)
            amplitude = amplitude.dropna()

            if len(amplitude) < self.window * 0.8:
                continue

            avg_amplitude = amplitude.mean()

            if not np.isnan(avg_amplitude) and not np.isinf(avg_amplitude):
                factor_values[ticker] = avg_amplitude

        if not factor_values:
            return pd.Series(dtype=float)

        factor_series = pd.Series(factor_values)
        ranks = factor_series.rank(pct=True)

        return ranks

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


@register('LiquidityFactor')
class LiquidityFactor(CrossSectionalFactor):
    """
    流动性因子 (Liquidity)。

    公式: L_t = 1/J * sum(Vol_i / |R_i|)
    即成交量与收益率绝对值的比值。

    方向: 负向因子 - 低流动性预期收益更高
    最优参数: J=1，年化收益1.64%，夏普0.25
    """

    direction = FactorDirection.NEGATIVE

    def __init__(
        self,
        name: str = "Liquidity",
        window: int = 1
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认1日)
        """
        super().__init__(name=name, window=window)
        self._params.update({'direction': 'NEGATIVE'})

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """计算全市场的流动性因子值。"""
        factor_values = {}

        for ticker, df in universe_data.items():
            if df is None or len(df) < self.window + 1:
                continue

            df = df.sort_index()
            if date is not None:
                df_until_date = df[df.index <= date]
            else:
                df_until_date = df

            if len(df_until_date) < self.window + 1:
                continue

            # 获取成交量
            vol_col = 'volume' if 'volume' in df_until_date.columns else 'S_DQ_VOLUME'
            if vol_col not in df_until_date.columns:
                continue

            volume = df_until_date[vol_col].iloc[-self.window:]

            # 计算收益率
            if 'returns' in df_until_date.columns:
                returns = df_until_date['returns'].iloc[-self.window:]
            else:
                close_col = 'close' if 'close' in df_until_date.columns else 'S_DQ_CLOSE'
                returns = df_until_date[close_col].pct_change().iloc[-self.window:]

            # 计算流动性 = Volume / |Returns|
            returns_abs = returns.abs().replace(0, np.nan)
            liquidity = (volume / returns_abs).replace([np.inf, -np.inf], np.nan)
            liquidity = liquidity.dropna()

            if len(liquidity) < 1:
                continue

            avg_liquidity = liquidity.mean()

            if not np.isnan(avg_liquidity) and not np.isinf(avg_liquidity):
                factor_values[ticker] = avg_liquidity

        if not factor_values:
            return pd.Series(dtype=float)

        factor_series = pd.Series(factor_values)

        # 负向因子: 低值排名高
        ranks = 1 - factor_series.rank(pct=True)

        return ranks

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


# 工厂函数：创建所有量价因子
def create_price_volume_factors(
    optimal_params: bool = True
) -> Dict[str, CrossSectionalFactor]:
    """
    创建所有量价类因子。

    Args:
        optimal_params: 是否使用最优参数 (报告中K=1时的最优J值)

    Returns:
        Dict[str, CrossSectionalFactor]: 因子名称到因子实例的映射
    """
    if optimal_params:
        return {
            'Volatility': VolatilityFactor(window=10),
            'CV': CVFactor(window=5),
            'Skewness': SkewnessFactor(window=10),
            'Kurtosis': KurtosisFactor(window=60),
            'Amplitude': AmplitudeFactor(window=5),
            'Liquidity': LiquidityFactor(window=1),
        }
    else:
        # 使用默认参数
        return {
            'Volatility': VolatilityFactor(),
            'CV': CVFactor(),
            'Skewness': SkewnessFactor(),
            'Kurtosis': KurtosisFactor(),
            'Amplitude': AmplitudeFactor(),
            'Liquidity': LiquidityFactor(),
        }
