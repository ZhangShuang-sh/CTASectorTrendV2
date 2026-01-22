#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Multi-Layer Signal Combiner

Implements 4-layer hierarchical signal fusion:
- L4: Sub-category level (equal weight within sub-category)
- L3: Category level (equal weight across sub-categories)
- L2: Scope level (Common + Idiosyncratic equally weighted)
- L1: Factor type level (weighted by factor_type_weights)

Final signal = Σ(type_weight * combined_signal_for_type)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from core.processors.config_loader import HierarchicalConfigLoader


@dataclass
class CombinedSignal:
    """组合信号数据类"""
    signal: float
    ts_contribution: float = 0.0
    xs_contribution: float = 0.0
    pair_contribution: float = 0.0
    common_contribution: float = 0.0
    idio_contribution: float = 0.0
    factor_breakdown: Dict[str, float] = field(default_factory=dict)
    pair: Optional[Tuple[str, str]] = None


class MultiLayerCombiner:
    """
    多层信号组合器

    四层融合逻辑:
    L4: 子类内等权 → L3: 大类内等权 → L2: Common+Idio等权 → L1: 类型加权

    使用方式:
        combiner = MultiLayerCombiner(config)
        combined = combiner.combine(factor_outputs, asset='RB')
    """

    def __init__(
        self,
        config: HierarchicalConfigLoader = None,
        factor_type_weights: Dict[str, float] = None
    ):
        """
        Args:
            config: 配置加载器
            factor_type_weights: L1 类型权重覆盖
        """
        if config is None:
            config = HierarchicalConfigLoader()
            config.load()

        self.config = config

        # L1 权重
        if factor_type_weights is None:
            factor_type_weights = config.get_factor_type_weights()

        self.factor_type_weights = factor_type_weights

    def combine(
        self,
        factor_outputs: Dict[str, Any],
        asset: str = None,
        sector: str = None,
        factor_metadata: Dict[str, Dict] = None
    ) -> CombinedSignal:
        """
        四层融合计算最终信号

        Args:
            factor_outputs: {factor_name: factor_value}
            asset: 资产代码 (用于区分 Idiosyncratic)
            sector: 行业代码
            factor_metadata: {factor_name: {'type', 'category', 'scope'}}

        Returns:
            CombinedSignal: 组合后的信号
        """
        if not factor_outputs:
            return CombinedSignal(signal=0.0)

        # 如果没有提供元数据，尝试从配置推断
        if factor_metadata is None:
            factor_metadata = self._infer_metadata(factor_outputs.keys())

        # L4 → L3: 按 category 分组并计算
        category_signals = self._combine_by_category(factor_outputs, factor_metadata)

        # L3 → L2: 按 scope (Common/Idio) 分组
        scope_signals = self._combine_by_scope(factor_outputs, factor_metadata)

        # L2 → L1: 按 factor_type 分组
        type_signals = self._combine_by_type(factor_outputs, factor_metadata)

        # L1 加权求和
        final_signal = 0.0
        ts_contrib = 0.0
        xs_contrib = 0.0
        pair_contrib = 0.0

        for factor_type, signal in type_signals.items():
            weight = self.factor_type_weights.get(factor_type, 0.0)
            final_signal += weight * signal

            if factor_type == 'time_series':
                ts_contrib = weight * signal
            elif factor_type == 'cross_sectional':
                xs_contrib = weight * signal
            elif factor_type == 'pair_trading':
                pair_contrib = weight * signal

        # 裁剪到 [-1, 1]
        final_signal = float(np.clip(final_signal, -1.0, 1.0))

        return CombinedSignal(
            signal=final_signal,
            ts_contribution=ts_contrib,
            xs_contribution=xs_contrib,
            pair_contribution=pair_contrib,
            common_contribution=scope_signals.get('common', 0.0),
            idio_contribution=scope_signals.get('idiosyncratic', 0.0),
            factor_breakdown={
                k: float(v) if isinstance(v, (int, float)) else 0.0
                for k, v in factor_outputs.items()
            }
        )

    def _combine_by_category(
        self,
        factor_outputs: Dict[str, Any],
        factor_metadata: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        L4 → L3: 按逻辑大类组合

        每个大类内的因子等权平均
        """
        category_groups: Dict[str, List[float]] = {}

        for name, value in factor_outputs.items():
            meta = factor_metadata.get(name, {})
            category = meta.get('category', 'unknown')

            if category not in category_groups:
                category_groups[category] = []

            # 提取数值
            if isinstance(value, (int, float)):
                category_groups[category].append(float(value))
            elif isinstance(value, pd.Series):
                # 取均值
                category_groups[category].append(float(value.mean()))
            elif isinstance(value, dict):
                # 字典取均值
                vals = [v for v in value.values() if isinstance(v, (int, float))]
                if vals:
                    category_groups[category].append(float(np.mean(vals)))

        # 每个大类内等权平均
        category_signals = {}
        for category, values in category_groups.items():
            if values:
                category_signals[category] = float(np.mean(values))
            else:
                category_signals[category] = 0.0

        return category_signals

    def _combine_by_scope(
        self,
        factor_outputs: Dict[str, Any],
        factor_metadata: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        L3 → L2: 按应用范围组合

        Common 和 Idiosyncratic 分别计算，然后等权
        """
        scope_groups: Dict[str, List[float]] = {
            'common': [],
            'idiosyncratic': []
        }

        for name, value in factor_outputs.items():
            meta = factor_metadata.get(name, {})
            scope = meta.get('scope', 'common')

            if scope not in scope_groups:
                scope_groups[scope] = []

            # 提取数值
            if isinstance(value, (int, float)):
                scope_groups[scope].append(float(value))
            elif isinstance(value, pd.Series):
                scope_groups[scope].append(float(value.mean()))
            elif isinstance(value, dict):
                vals = [v for v in value.values() if isinstance(v, (int, float))]
                if vals:
                    scope_groups[scope].append(float(np.mean(vals)))

        # 每个范围内等权平均
        scope_signals = {}
        for scope, values in scope_groups.items():
            if values:
                scope_signals[scope] = float(np.mean(values))
            else:
                scope_signals[scope] = 0.0

        return scope_signals

    def _combine_by_type(
        self,
        factor_outputs: Dict[str, Any],
        factor_metadata: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        L2 → L1: 按因子类型组合

        每个类型内的因子等权平均
        """
        type_groups: Dict[str, List[float]] = {
            'time_series': [],
            'cross_sectional': [],
            'pair_trading': []
        }

        for name, value in factor_outputs.items():
            meta = factor_metadata.get(name, {})
            factor_type = meta.get('type') or meta.get('factor_type', 'time_series')

            if factor_type not in type_groups:
                type_groups[factor_type] = []

            # 提取数值
            if isinstance(value, (int, float)):
                type_groups[factor_type].append(float(value))
            elif isinstance(value, pd.Series):
                type_groups[factor_type].append(float(value.mean()))
            elif isinstance(value, dict):
                vals = [v for v in value.values() if isinstance(v, (int, float))]
                if vals:
                    type_groups[factor_type].append(float(np.mean(vals)))

        # 每个类型内等权平均
        type_signals = {}
        for factor_type, values in type_groups.items():
            if values:
                type_signals[factor_type] = float(np.mean(values))
            else:
                type_signals[factor_type] = 0.0

        return type_signals

    def _infer_metadata(self, factor_names: List[str]) -> Dict[str, Dict]:
        """从配置推断因子元数据"""
        metadata = {}

        for name in factor_names:
            fc = self.config.get_factor_config(name)
            if fc:
                metadata[name] = {
                    'type': fc.get('factor_type', 'time_series'),
                    'category': fc.get('category', 'unknown'),
                    'scope': 'common'  # 从配置获取的都是 common
                }
            else:
                # 默认
                metadata[name] = {
                    'type': 'time_series',
                    'category': 'unknown',
                    'scope': 'common'
                }

        return metadata

    def combine_pair_signals(
        self,
        pair_factor_outputs: Dict[Tuple[str, str], float],
        single_factor_outputs: Dict[str, float] = None
    ) -> Dict[Tuple[str, str], CombinedSignal]:
        """
        组合配对信号

        将配对因子输出和单资产因子输出组合

        Args:
            pair_factor_outputs: {(asset1, asset2): signal}
            single_factor_outputs: {asset: signal}

        Returns:
            {(asset1, asset2): CombinedSignal}
        """
        results = {}

        for pair, pair_signal in pair_factor_outputs.items():
            asset1, asset2 = pair

            # 获取单资产信号
            signal1 = single_factor_outputs.get(asset1, 0.0) if single_factor_outputs else 0.0
            signal2 = single_factor_outputs.get(asset2, 0.0) if single_factor_outputs else 0.0

            # 组合信号：配对信号 + (asset1信号 - asset2信号) / 2
            single_diff = (signal1 - signal2) / 2
            combined = pair_signal * 0.7 + single_diff * 0.3  # 配对信号权重更高

            results[pair] = CombinedSignal(
                signal=float(np.clip(combined, -1.0, 1.0)),
                pair_contribution=float(pair_signal),
                ts_contribution=float(single_diff),
                pair=pair
            )

        return results

    def __repr__(self) -> str:
        return f"MultiLayerCombiner(weights={self.factor_type_weights})"


# 便捷函数
def create_combiner(
    config: HierarchicalConfigLoader = None,
    factor_type_weights: Dict[str, float] = None
) -> MultiLayerCombiner:
    """创建多层组合器"""
    return MultiLayerCombiner(
        config=config,
        factor_type_weights=factor_type_weights
    )
