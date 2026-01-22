#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计语言模型(SLM)择时因子

来源: 广发证券《基于统计语言模型（SLM）的择时交易研究》(2014年1月14日)
系列: 另类交易策略之十三

理论背景:
- 借鉴自然语言处理中的N-gram统计语言模型
- 将价格涨跌符号化为符号序列
- 根据历史符号串的条件概率预测未来涨跌

核心逻辑：
1. 将价格序列符号化：上涨=2，下跌=1
2. 构建N-gram语料库，统计各符号串出现频率
3. 根据最近N-1天的符号串，计算下一天涨跌的条件概率
4. 概率较大者为预测结果

原报告参数：
- 模型阶数N=6为最优
- 止损幅度1%
- 在上证指数上年化收益80.3%，胜率46.1%
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from collections import defaultdict

from core.factors.time_series.base import TimeSeriesFactorBase


class SLMTimingFactor(TimeSeriesFactorBase):
    """
    统计语言模型择时因子

    基于N-gram统计语言模型，将价格涨跌符号化后，
    根据历史符号串的条件概率预测未来涨跌方向。

    信号输出：
    - +1: 预测上涨，做多
    - -1: 预测下跌，做空
    - 0: 无信号（数据不足或概率相等）
    """

    def __init__(
        self,
        model_order: int = 6,
        min_corpus_size: int = 200,
        laplace_smoothing: bool = True,
        smoothing_alpha: float = 1.0,
        name: str = None
    ):
        """
        Args:
            model_order: 模型阶数N，即使用最近N-1天的涨跌模式预测第N天
            min_corpus_size: 最小语料库大小，少于此值不产生信号
            laplace_smoothing: 是否使用拉普拉斯平滑
            smoothing_alpha: 拉普拉斯平滑系数
        """
        # 窗口需要足够长以建立语料库并计算信号
        window = max(min_corpus_size + model_order, 300)

        if name is None:
            name = f"SLMTiming_{model_order}"

        super().__init__(name=name, window=window)

        self.model_order = model_order
        self.min_corpus_size = min_corpus_size
        self.laplace_smoothing = laplace_smoothing
        self.smoothing_alpha = smoothing_alpha

        # 更新参数字典
        self._params.update({
            'model_order': model_order,
            'min_corpus_size': min_corpus_size,
            'laplace_smoothing': laplace_smoothing,
            'smoothing_alpha': smoothing_alpha
        })

    def _symbolize(self, prices: pd.Series) -> pd.Series:
        """
        将价格序列转化为符号序列

        符号规则：
        - 1: 下跌 (当日收盘价 < 前一日收盘价)
        - 2: 上涨或持平 (当日收盘价 >= 前一日收盘价)

        Args:
            prices: 收盘价序列

        Returns:
            符号序列
        """
        # 计算涨跌
        returns = prices.diff()
        # 转化为符号：下跌=1，上涨/持平=2
        symbols = pd.Series(index=prices.index, dtype=int)
        symbols[returns < 0] = 1
        symbols[returns >= 0] = 2
        symbols.iloc[0] = np.nan  # 第一个值无法判断涨跌
        return symbols

    def _build_corpus(self, symbols: pd.Series) -> Dict[str, int]:
        """
        构建N-gram语料库

        统计所有长度为model_order的符号串出现次数

        Args:
            symbols: 符号序列

        Returns:
            Dict: 符号串 -> 出现次数
        """
        corpus = defaultdict(int)

        # 转换为字符串列表
        symbols_clean = symbols.dropna().astype(int).astype(str).tolist()

        # 滑动窗口提取所有N-gram
        for i in range(len(symbols_clean) - self.model_order + 1):
            ngram = ''.join(symbols_clean[i:i + self.model_order])
            corpus[ngram] += 1

        return dict(corpus)

    def _get_conditional_prob(
        self,
        corpus: Dict[str, int],
        history: str,
        next_symbol: str
    ) -> float:
        """
        计算条件概率 P(next_symbol | history)

        Args:
            corpus: N-gram语料库
            history: 历史符号串（长度N-1）
            next_symbol: 下一个符号（'1'或'2'）

        Returns:
            条件概率
        """
        # 完整的N-gram
        full_ngram = history + next_symbol

        # 统计history开头的所有N-gram的总数
        history_count = 0
        ngram_count = 0

        for ngram, count in corpus.items():
            if ngram.startswith(history):
                history_count += count
                if ngram == full_ngram:
                    ngram_count = count

        if self.laplace_smoothing:
            # 拉普拉斯平滑：(count + alpha) / (total + alpha * V)
            # V = 2 (符号种类数)
            vocab_size = 2
            prob = (ngram_count + self.smoothing_alpha) / \
                   (history_count + self.smoothing_alpha * vocab_size)
        else:
            # 无平滑
            if history_count == 0:
                prob = 0.5  # 未见过的历史，返回均匀分布
            else:
                prob = ngram_count / history_count

        return prob

    def _predict(self, corpus: Dict[str, int], history: str) -> tuple:
        """
        基于语料库和历史符号串预测下一个交易日的涨跌

        Args:
            corpus: N-gram语料库
            history: 最近N-1天的符号串

        Returns:
            Tuple[int, float]: (预测信号, 概率差)
            - 信号: +1=做多, -1=做空, 0=无信号
            - 概率差: |P(涨) - P(跌)|
        """
        # 计算上涨和下跌的条件概率
        prob_up = self._get_conditional_prob(corpus, history, '2')
        prob_down = self._get_conditional_prob(corpus, history, '1')

        # 根据概率大小决定方向
        prob_diff = prob_up - prob_down

        if prob_up > prob_down:
            return 1, abs(prob_diff)
        elif prob_up < prob_down:
            return -1, abs(prob_diff)
        else:
            return 0, 0.0

    def calculate(self, data: pd.DataFrame) -> float:
        """
        计算当前信号

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            float: 信号值 (+1=做多, -1=做空, 0=无信号)
        """
        if len(data) < self.min_corpus_size + self.model_order:
            return 0.0

        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return 0.0

        prices = data[close_col]

        # 符号化
        symbols = self._symbolize(prices)

        # 用于建立语料库的数据（不包括最后一天，因为要预测最后一天）
        corpus_symbols = symbols.iloc[:-1]

        # 检查语料库大小
        valid_symbols = corpus_symbols.dropna()
        if len(valid_symbols) < self.min_corpus_size:
            return 0.0

        # 构建语料库
        corpus = self._build_corpus(corpus_symbols)

        # 获取最近N-1天的符号串（用于预测）
        recent_symbols = valid_symbols.iloc[-(self.model_order - 1):]
        history = ''.join(recent_symbols.astype(int).astype(str).tolist())

        # 预测
        signal, _ = self._predict(corpus, history)

        return float(signal)

    def compute_series(self, data: pd.DataFrame) -> pd.Series:
        """
        计算完整的因子序列（用于回测）

        使用滚动窗口方式：
        1. 对于每个时间点t，使用t之前的数据建立语料库
        2. 基于t-N+1到t-1的符号串预测t的涨跌

        Args:
            data: 必须包含 'close' 或 'S_DQ_CLOSE' 列

        Returns:
            pd.Series: 信号序列
        """
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return pd.Series(dtype=float)

        prices = data[close_col]
        n = len(prices)

        # 初始化信号序列
        signals = pd.Series(index=data.index, dtype=float)
        signals[:] = 0.0

        # 符号化整个序列
        all_symbols = self._symbolize(prices)

        # 从最小语料库大小之后开始计算信号
        start_idx = self.min_corpus_size + self.model_order

        for i in range(start_idx, n):
            # 使用i之前的数据建立语料库
            corpus_symbols = all_symbols.iloc[:i]
            valid_symbols = corpus_symbols.dropna()

            if len(valid_symbols) < self.min_corpus_size:
                continue

            # 构建语料库
            corpus = self._build_corpus(corpus_symbols)

            # 获取预测所需的历史符号串
            # 预测第i天，需要第i-N+1到i-1天的符号
            history_symbols = valid_symbols.iloc[-(self.model_order - 1):]
            if len(history_symbols) < self.model_order - 1:
                continue

            history = ''.join(history_symbols.astype(int).astype(str).tolist())

            # 预测
            signal, _ = self._predict(corpus, history)
            signals.iloc[i] = float(signal)

        return signals

    def get_prediction_probs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        获取详细的预测概率（用于分析）

        Returns:
            DataFrame: 包含 prob_up, prob_down, signal 列
        """
        close_col = 'close' if 'close' in data.columns else 'S_DQ_CLOSE'
        if close_col not in data.columns:
            return pd.DataFrame()

        prices = data[close_col]
        n = len(prices)

        # 初始化结果
        results = pd.DataFrame(
            index=data.index,
            columns=['prob_up', 'prob_down', 'signal']
        )
        results[:] = np.nan

        # 符号化整个序列
        all_symbols = self._symbolize(prices)

        start_idx = self.min_corpus_size + self.model_order

        for i in range(start_idx, n):
            corpus_symbols = all_symbols.iloc[:i]
            valid_symbols = corpus_symbols.dropna()

            if len(valid_symbols) < self.min_corpus_size:
                continue

            corpus = self._build_corpus(corpus_symbols)

            history_symbols = valid_symbols.iloc[-(self.model_order - 1):]
            if len(history_symbols) < self.model_order - 1:
                continue

            history = ''.join(history_symbols.astype(int).astype(str).tolist())

            # 计算概率
            prob_up = self._get_conditional_prob(corpus, history, '2')
            prob_down = self._get_conditional_prob(corpus, history, '1')

            results.iloc[i, 0] = prob_up
            results.iloc[i, 1] = prob_down
            results.iloc[i, 2] = 1 if prob_up > prob_down else (-1 if prob_up < prob_down else 0)

        return results


# 便捷工厂函数
def create_slm_factor(
    model_order: int = 6,
    min_corpus_size: int = 200,
    **kwargs
) -> SLMTimingFactor:
    """
    创建SLM择时因子

    Args:
        model_order: 模型阶数（默认6，原报告最优参数）
        min_corpus_size: 最小语料库大小
        **kwargs: 其他参数

    Returns:
        SLMTimingFactor实例
    """
    return SLMTimingFactor(
        model_order=model_order,
        min_corpus_size=min_corpus_size,
        **kwargs
    )


def create_short_term_slm(**kwargs) -> SLMTimingFactor:
    """
    创建短周期SLM因子（N=4）

    适合数据量较少的品种或快速反应
    """
    defaults = {
        'model_order': 4,
        'min_corpus_size': 150,
    }
    defaults.update(kwargs)
    return SLMTimingFactor(**defaults)


def create_optimal_slm(**kwargs) -> SLMTimingFactor:
    """
    创建原报告最优参数SLM因子（N=6）

    原报告通过样本内优化确定的最优参数
    """
    defaults = {
        'model_order': 6,
        'min_corpus_size': 200,
    }
    defaults.update(kwargs)
    return SLMTimingFactor(**defaults)


def create_long_term_slm(**kwargs) -> SLMTimingFactor:
    """
    创建长周期SLM因子（N=8）

    考虑更长的历史模式，需要更多数据
    """
    defaults = {
        'model_order': 8,
        'min_corpus_size': 300,
    }
    defaults.update(kwargs)
    return SLMTimingFactor(**defaults)
