#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - CTA Momentum Cross-Sectional Factors

CTA动量类截面因子实现。

基于中信期货研究报告《CTA风格因子手册（二）：动量类因子》实现的5个因子:
- CrossSectionalMomentumFactor: 截面动量因子
- TimeSeriesMomentumFactor: 时序动量因子
- CompositeMomentumFactor: 复合动量因子
- BIASFactor: 乖离率因子
- TrendStrengthFactor: 趋势强度因子

交易模式: Close-to-Close (T日信号，T日收盘成交)
- T日临近收盘计算因子值
- T日收盘价成交建仓
- 收益来自 T日收盘 → T+K日收盘

所有因子均为正向指标: 因子值越大，预期收益越高，做多高值做空低值。

Source: V1 factors/cross_sectional/momentum_citic.py (100% logic preserved)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from enum import Enum

from core.factors.cross_sectional.base import CrossSectionalFactor
from core.factors.registry import register


class MomentumDirection(Enum):
    """动量因子方向枚举"""
    POSITIVE = 1   # 正向因子：做多高值，做空低值


@register('CrossSectionalMomentumFactor')
class CrossSectionalMomentumFactor(CrossSectionalFactor):
    """
    截面动量因子 (Cross-Sectional Momentum)。

    公式: Mom_t = (P_t - P_{t-J}) / P_{t-J}

    策略规则:
    - 对所有品种的因子值从高到低排序
    - 做多: 因子排名靠前的品种 (前20%)
    - 做空: 因子排名靠后的品种 (后20%)

    方向: 正向因子 - 高动量预期收益更高
    最优参数: J=240, K=1, 年化收益5.59%, 夏普0.57

    注意: 计算T日因子只使用T日及之前的数据，严禁未来函数。
    """

    direction = MomentumDirection.POSITIVE

    def __init__(
        self,
        name: str = "CrossSectionalMomentum",
        window: int = 240,
        long_pct: float = 0.2,
        short_pct: float = 0.2
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认240日)
            long_pct: 多头比例 (默认20%)
            short_pct: 空头比例 (默认20%)
        """
        super().__init__(name=name, window=window)
        self.long_pct = long_pct
        self.short_pct = short_pct
        self._params.update({
            'direction': 'POSITIVE',
            'long_pct': long_pct,
            'short_pct': short_pct,
            'strategy_type': 'cross_sectional'
        })

    def _calculate_momentum(self, close_series: pd.Series) -> Optional[float]:
        """
        计算单个品种的动量值。

        Args:
            close_series: 收盘价序列，长度至少为 window+1

        Returns:
            动量值 (P_t - P_{t-J}) / P_{t-J}，如果数据不足返回None
        """
        if len(close_series) < self.window + 1:
            return None

        p_t = close_series.iloc[-1]  # T日收盘价
        p_t_j = close_series.iloc[-(self.window + 1)]  # T-J日收盘价

        if p_t_j == 0 or np.isnan(p_t_j) or np.isnan(p_t):
            return None

        momentum = (p_t - p_t_j) / p_t_j
        return momentum

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """
        计算全市场的截面动量因子值。

        Args:
            universe_data: 全市场数据字典
                Key: Ticker (如 'RB.SHF')
                Value: 该品种截至当前日期的历史数据 DataFrame
            date: 当前回测截面的日期 (T日)

        Returns:
            pd.Series: 归一化后的动量排名分数 [0, 1]
                       高值 = 做多，低值 = 做空
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

            # 获取收盘价列
            close_col = 'close' if 'close' in df_until_date.columns else 'S_DQ_CLOSE'
            if close_col not in df_until_date.columns:
                continue

            close = df_until_date[close_col]
            momentum = self._calculate_momentum(close)

            if momentum is not None and not np.isnan(momentum) and not np.isinf(momentum):
                factor_values[ticker] = momentum

        if not factor_values:
            return pd.Series(dtype=float)

        factor_series = pd.Series(factor_values)

        # 正向因子: 高值排名高 (ascending=False for rank means high value gets high rank)
        ranks = factor_series.rank(pct=True)

        return ranks

    def get_long_short_threshold(self) -> Tuple[float, float]:
        """返回多空阈值"""
        return (1 - self.long_pct, self.short_pct)

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


@register('TimeSeriesMomentumFactor')
class TimeSeriesMomentumFactor(CrossSectionalFactor):
    """
    时序动量因子 (Time-Series Momentum)。

    公式: Mom_t = (P_t - P_{t-J}) / P_{t-J}

    策略规则:
    - 根据因子值的正负直接决定方向
    - 做多: 因子值为正的品种 (Mom_t > 0)
    - 做空: 因子值为负的品种 (Mom_t < 0)

    方向: 正向因子
    最优参数: J=240, K=1, 年化收益4.29%, 夏普0.47

    注意: 与截面动量使用相同公式，区别在于信号生成规则。
    """

    direction = MomentumDirection.POSITIVE

    def __init__(
        self,
        name: str = "TimeSeriesMomentum",
        window: int = 240
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认240日)
        """
        super().__init__(name=name, window=window)
        self._params.update({
            'direction': 'POSITIVE',
            'strategy_type': 'time_series'
        })

    def _calculate_momentum(self, close_series: pd.Series) -> Optional[float]:
        """计算单个品种的动量值。"""
        if len(close_series) < self.window + 1:
            return None

        p_t = close_series.iloc[-1]
        p_t_j = close_series.iloc[-(self.window + 1)]

        if p_t_j == 0 or np.isnan(p_t_j) or np.isnan(p_t):
            return None

        momentum = (p_t - p_t_j) / p_t_j
        return momentum

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """
        计算全市场的时序动量因子值。

        返回值含义:
        - 正值: 做多该品种
        - 负值: 做空该品种
        - 绝对值大小反映信号强度

        Returns:
            pd.Series: 原始动量值 (未归一化)
                       正值 = 做多，负值 = 做空
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

            close_col = 'close' if 'close' in df_until_date.columns else 'S_DQ_CLOSE'
            if close_col not in df_until_date.columns:
                continue

            close = df_until_date[close_col]
            momentum = self._calculate_momentum(close)

            if momentum is not None and not np.isnan(momentum) and not np.isinf(momentum):
                factor_values[ticker] = momentum

        if not factor_values:
            return pd.Series(dtype=float)

        # 时序动量返回原始值，正负决定方向
        return pd.Series(factor_values)

    def generate_signal(self, factor_value: float) -> int:
        """
        根据因子值生成交易信号。

        Args:
            factor_value: 因子值

        Returns:
            1 = 做多, -1 = 做空, 0 = 无信号
        """
        if factor_value > 0:
            return 1
        elif factor_value < 0:
            return -1
        return 0

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


@register('CompositeMomentumFactor')
class CompositeMomentumFactor(CrossSectionalFactor):
    """
    复合动量因子 (Composite Momentum)。

    公式: Mom_t = (P_t - P_{t-J}) / P_{t-J}

    策略规则 (时序筛选 + 截面排序):
    1. 时序筛选: 按因子值正负分为两组
       - 正组合: Mom_t > 0 的所有品种
       - 负组合: Mom_t < 0 的所有品种
    2. 截面排序: 在各组内部进行排序
       - 做多: 正组合中，因子值降序排名前40%的品种
       - 做空: 负组合中，因子值降序排名后40%的品种

    方向: 正向因子
    最优参数: J=240, K=1, 年化收益7.99%, 夏普0.56

    注意: 复合策略是表现最好的动量策略。
    """

    direction = MomentumDirection.POSITIVE

    def __init__(
        self,
        name: str = "CompositeMomentum",
        window: int = 240,
        group_pct: float = 0.4
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认240日)
            group_pct: 组内多空比例 (默认40%)
        """
        super().__init__(name=name, window=window)
        self.group_pct = group_pct
        self._params.update({
            'direction': 'POSITIVE',
            'group_pct': group_pct,
            'strategy_type': 'composite'
        })

    def _calculate_momentum(self, close_series: pd.Series) -> Optional[float]:
        """计算单个品种的动量值。"""
        if len(close_series) < self.window + 1:
            return None

        p_t = close_series.iloc[-1]
        p_t_j = close_series.iloc[-(self.window + 1)]

        if p_t_j == 0 or np.isnan(p_t_j) or np.isnan(p_t):
            return None

        momentum = (p_t - p_t_j) / p_t_j
        return momentum

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """
        计算全市场的复合动量因子值。

        Returns:
            pd.Series: 复合动量信号
                       1.0 = 强烈做多 (正组合中排名靠前)
                       0.5 = 中等做多 (正组合中排名靠后)
                       -0.5 = 中等做空 (负组合中排名靠前)
                       -1.0 = 强烈做空 (负组合中排名靠后)
                       0.0 = 无信号
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

            close_col = 'close' if 'close' in df_until_date.columns else 'S_DQ_CLOSE'
            if close_col not in df_until_date.columns:
                continue

            close = df_until_date[close_col]
            momentum = self._calculate_momentum(close)

            if momentum is not None and not np.isnan(momentum) and not np.isinf(momentum):
                factor_values[ticker] = momentum

        if not factor_values:
            return pd.Series(dtype=float)

        momentum_series = pd.Series(factor_values)

        # Step 1: 时序筛选 - 分为正负两组
        positive_group = momentum_series[momentum_series > 0]
        negative_group = momentum_series[momentum_series < 0]

        result = pd.Series(0.0, index=momentum_series.index)

        # Step 2: 正组合内截面排序
        if len(positive_group) > 0:
            pos_ranks = positive_group.rank(pct=True, ascending=True)
            # 高值排名靠前 -> 做多
            long_threshold = 1 - self.group_pct
            for ticker in pos_ranks.index:
                if pos_ranks[ticker] >= long_threshold:
                    # 强烈做多信号，根据排名给予 [0.5, 1.0] 的分数
                    result[ticker] = 0.5 + 0.5 * (pos_ranks[ticker] - long_threshold) / self.group_pct
                else:
                    # 正组合中但未入选，给予小正值
                    result[ticker] = 0.25 * pos_ranks[ticker]

        # Step 3: 负组合内截面排序
        if len(negative_group) > 0:
            neg_ranks = negative_group.rank(pct=True, ascending=True)
            # 低值排名靠后 -> 做空
            short_threshold = self.group_pct
            for ticker in neg_ranks.index:
                if neg_ranks[ticker] <= short_threshold:
                    # 强烈做空信号，根据排名给予 [-1.0, -0.5] 的分数
                    result[ticker] = -0.5 - 0.5 * (short_threshold - neg_ranks[ticker]) / self.group_pct
                else:
                    # 负组合中但未入选，给予小负值
                    result[ticker] = -0.25 * (1 - neg_ranks[ticker])

        return result

    def get_long_short_tickers(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None
    ) -> Tuple[List[str], List[str]]:
        """
        获取做多和做空的品种列表。

        Returns:
            (long_tickers, short_tickers): 做多品种列表, 做空品种列表
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

            close_col = 'close' if 'close' in df_until_date.columns else 'S_DQ_CLOSE'
            if close_col not in df_until_date.columns:
                continue

            close = df_until_date[close_col]
            momentum = self._calculate_momentum(close)

            if momentum is not None and not np.isnan(momentum) and not np.isinf(momentum):
                factor_values[ticker] = momentum

        if not factor_values:
            return [], []

        momentum_series = pd.Series(factor_values)

        # 分组
        positive_group = momentum_series[momentum_series > 0].sort_values(ascending=False)
        negative_group = momentum_series[momentum_series < 0].sort_values(ascending=True)

        # 选择
        n_long = max(1, int(len(positive_group) * self.group_pct))
        n_short = max(1, int(len(negative_group) * self.group_pct))

        long_tickers = positive_group.head(n_long).index.tolist()
        short_tickers = negative_group.head(n_short).index.tolist()

        return long_tickers, short_tickers

    # Backward compatibility alias
    def compute_all(
        self,
        date: pd.Timestamp,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Backward compatible method name"""
        return self.calculate(universe_data, date=date)


@register('BIASFactor')
class BIASFactor(CrossSectionalFactor):
    """
    乖离率因子 (BIAS)。

    公式: BIAS_t = (P_t - P_bar) / (sigma(R) * P_{t-J})

    其中:
    - P_t: T日收盘价
    - P_bar: 过去J天的均价 = 1/J * sum(P_i)
    - sigma(R): 过去J天收益率的标准差
    - P_{t-J}: T-J日收盘价

    策略规则:
    - 对所有品种的因子值从高到低排序
    - 做多: 因子排名靠前的品种 (向上突破均价较多)
    - 做空: 因子排名靠后的品种 (向下突破均价较多)

    方向: 正向因子
    最优参数: J=20, K=1, 年化收益2.96%, 夏普0.34
    """

    direction = MomentumDirection.POSITIVE

    def __init__(
        self,
        name: str = "BIAS",
        window: int = 20,
        long_pct: float = 0.2,
        short_pct: float = 0.2
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认20日)
            long_pct: 多头比例 (默认20%)
            short_pct: 空头比例 (默认20%)
        """
        super().__init__(name=name, window=window)
        self.long_pct = long_pct
        self.short_pct = short_pct
        self._params.update({
            'direction': 'POSITIVE',
            'long_pct': long_pct,
            'short_pct': short_pct
        })

    def _calculate_bias(self, close_series: pd.Series) -> Optional[float]:
        """
        计算单个品种的乖离率值。

        Args:
            close_series: 收盘价序列，长度至少为 window+1

        Returns:
            乖离率值，如果数据不足返回None
        """
        if len(close_series) < self.window + 1:
            return None

        # 取最近 window+1 个数据点
        prices = close_series.iloc[-(self.window + 1):]

        p_t = prices.iloc[-1]  # T日收盘价
        p_t_j = prices.iloc[0]  # T-J日收盘价

        if p_t_j == 0 or np.isnan(p_t_j) or np.isnan(p_t):
            return None

        # 计算过去J天的均价 (从T-J+1到T)
        p_bar = prices.iloc[1:].mean()

        # 计算收益率序列
        returns = prices.pct_change().dropna()
        if len(returns) < 2:
            return None

        # 计算收益率标准差
        sigma_r = returns.std()
        if sigma_r == 0 or np.isnan(sigma_r):
            return None

        # BIAS = (P_t - P_bar) / (sigma(R) * P_{t-J})
        bias = (p_t - p_bar) / (sigma_r * p_t_j)

        return bias

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """
        计算全市场的乖离率因子值。

        Args:
            universe_data: 全市场数据字典
            date: 当前回测截面的日期

        Returns:
            pd.Series: 归一化后的乖离率排名分数 [0, 1]
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

            close_col = 'close' if 'close' in df_until_date.columns else 'S_DQ_CLOSE'
            if close_col not in df_until_date.columns:
                continue

            close = df_until_date[close_col]
            bias = self._calculate_bias(close)

            if bias is not None and not np.isnan(bias) and not np.isinf(bias):
                factor_values[ticker] = bias

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


@register('TrendStrengthFactor')
class TrendStrengthFactor(CrossSectionalFactor):
    """
    趋势强度因子 (Trend Strength)。

    公式: TS_t = (P_t - P_{t-J}) / sum(|P_i - P_{i-1}|)

    其中:
    - 分子 (P_t - P_{t-J}): 收盘价的"位移" (净变化)
    - 分母 sum(|P_i - P_{i-1}|): 收盘价的"路程" (绝对变化之和)

    物理意义:
    - TS_t ≈ 1: 价格持续上涨，趋势强
    - TS_t ≈ -1: 价格持续下跌，趋势强
    - TS_t ≈ 0: 价格震荡，无明显趋势

    策略规则:
    - 对所有品种的因子值从高到低排序
    - 做多: 趋势强度较大的品种 (排名靠前)
    - 做空: 趋势强度较小的品种 (排名靠后)

    方向: 正向因子
    最优参数: J=20, K=1, 年化收益0.86%, 夏普0.12
    """

    direction = MomentumDirection.POSITIVE

    def __init__(
        self,
        name: str = "TrendStrength",
        window: int = 20,
        long_pct: float = 0.2,
        short_pct: float = 0.2
    ):
        """
        Args:
            name: 因子名称
            window: 回溯窗口J (默认20日)
            long_pct: 多头比例 (默认20%)
            short_pct: 空头比例 (默认20%)
        """
        super().__init__(name=name, window=window)
        self.long_pct = long_pct
        self.short_pct = short_pct
        self._params.update({
            'direction': 'POSITIVE',
            'long_pct': long_pct,
            'short_pct': short_pct
        })

    def _calculate_trend_strength(self, close_series: pd.Series) -> Optional[float]:
        """
        计算单个品种的趋势强度值。

        Args:
            close_series: 收盘价序列，长度至少为 window+1

        Returns:
            趋势强度值，如果数据不足返回None
        """
        if len(close_series) < self.window + 1:
            return None

        # 取最近 window+1 个数据点
        prices = close_series.iloc[-(self.window + 1):]

        p_t = prices.iloc[-1]  # T日收盘价
        p_t_j = prices.iloc[0]  # T-J日收盘价

        if np.isnan(p_t_j) or np.isnan(p_t):
            return None

        # 计算位移 (净变化)
        displacement = p_t - p_t_j

        # 计算路程 (绝对变化之和)
        price_diffs = prices.diff().dropna().abs()
        total_path = price_diffs.sum()

        if total_path == 0 or np.isnan(total_path):
            return None

        # TS = 位移 / 路程
        trend_strength = displacement / total_path

        return trend_strength

    def calculate(
        self,
        universe_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        **kwargs
    ) -> pd.Series:
        """
        计算全市场的趋势强度因子值。

        Args:
            universe_data: 全市场数据字典
            date: 当前回测截面的日期

        Returns:
            pd.Series: 归一化后的趋势强度排名分数 [0, 1]
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

            close_col = 'close' if 'close' in df_until_date.columns else 'S_DQ_CLOSE'
            if close_col not in df_until_date.columns:
                continue

            close = df_until_date[close_col]
            ts = self._calculate_trend_strength(close)

            if ts is not None and not np.isnan(ts) and not np.isinf(ts):
                factor_values[ticker] = ts

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


# 工厂函数：创建所有动量因子
def create_momentum_factors(
    optimal_params: bool = True
) -> Dict[str, CrossSectionalFactor]:
    """
    创建所有动量类因子。

    Args:
        optimal_params: 是否使用最优参数 (报告中K=1时的最优J值)

    Returns:
        Dict[str, CrossSectionalFactor]: 因子名称到因子实例的映射
    """
    if optimal_params:
        return {
            'CrossSectionalMomentum': CrossSectionalMomentumFactor(window=240),
            'TimeSeriesMomentum': TimeSeriesMomentumFactor(window=240),
            'CompositeMomentum': CompositeMomentumFactor(window=240),
            'BIAS': BIASFactor(window=20),
            'TrendStrength': TrendStrengthFactor(window=20),
        }
    else:
        # 使用默认参数
        return {
            'CrossSectionalMomentum': CrossSectionalMomentumFactor(),
            'TimeSeriesMomentum': TimeSeriesMomentumFactor(),
            'CompositeMomentum': CompositeMomentumFactor(),
            'BIAS': BIASFactor(),
            'TrendStrength': TrendStrengthFactor(),
        }


# 便捷别名
CSMomentum = CrossSectionalMomentumFactor
TSMomentum = TimeSeriesMomentumFactor
ComboMomentum = CompositeMomentumFactor
BIAS = BIASFactor
TrendStr = TrendStrengthFactor
