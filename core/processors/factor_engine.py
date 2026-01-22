#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Factor Compute Engine

Pure orchestration layer for factor computation:
- No factor logic (factors are completely decoupled)
- Dispatches to appropriate factor instances based on type
- Handles data preparation and result collection

This is the main entry point for computing factors.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from core.factors.base import FactorBase, FactorType
from core.factors.registry import FactorRegistry, get_registry
from core.processors.config_loader import HierarchicalConfigLoader
from core.processors.param_resolver import ParameterResolver
from core.processors.normalizer import SignalNormalizer
from core.processors.combiner import MultiLayerCombiner, CombinedSignal


@dataclass
class FactorOutput:
    """因子输出数据类"""
    name: str
    factor_type: FactorType
    values: Union[float, pd.Series, Dict]
    raw_values: Any = None
    normalized: bool = False
    metadata: Dict = field(default_factory=dict)


class FactorComputeEngine:
    """
    因子计算引擎 (V2版本)

    纯编排层，不包含任何因子逻辑:
    1. 从配置加载因子定义
    2. 通过 Registry 实例化因子
    3. 通过 ParameterResolver 解析参数
    4. 调用因子的 calculate() 方法
    5. 通过 SignalNormalizer 正则化
    6. 通过 MultiLayerCombiner 组合信号

    使用方式:
        engine = FactorComputeEngine()
        outputs = engine.compute_all(data, date, asset='RB', sector='煤焦钢矿')
        combined = engine.combine_signals(outputs)
    """

    def __init__(
        self,
        config: HierarchicalConfigLoader = None,
        registry: FactorRegistry = None,
        normalizer: SignalNormalizer = None,
        combiner: MultiLayerCombiner = None
    ):
        """
        Args:
            config: 配置加载器
            registry: 因子注册表
            normalizer: 信号正则化器
            combiner: 多层组合器
        """
        if config is None:
            config = HierarchicalConfigLoader()
            config.load()

        self.config = config
        self.registry = registry or get_registry()
        self.param_resolver = ParameterResolver(config)
        self.normalizer = normalizer or SignalNormalizer()
        self.combiner = combiner or MultiLayerCombiner(config)

        # 注册配置中的因子
        self._register_configured_factors()

    def _register_configured_factors(self) -> None:
        """从配置注册因子到 Registry"""
        all_factors = self.config.get_common_factors()

        for fc in all_factors:
            name = fc.get('name')
            class_path = fc.get('class')

            if name and class_path and name not in self.registry:
                self.registry.register_from_path(
                    name=name,
                    class_path=class_path,
                    metadata={
                        'category': fc.get('category'),
                        'default_params': fc.get('default_params', {})
                    }
                )

    def compute_all(
        self,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp = None,
        asset: str = None,
        sector: str = None,
        factor_types: List[str] = None,
        normalize: bool = True
    ) -> Dict[str, FactorOutput]:
        """
        计算所有适用因子

        Args:
            data: {asset_code: DataFrame} 市场数据
            date: 当前日期
            asset: 当前资产 (用于参数解析)
            sector: 当前行业 (用于参数解析)
            factor_types: 因子类型过滤 (None = 全部)
            normalize: 是否正则化输出

        Returns:
            {factor_name: FactorOutput}
        """
        outputs = {}

        # 获取适用因子
        applicable_factors = self.param_resolver.get_applicable_factors(
            factor_type=None,  # 获取所有类型
            asset=asset,
            sector=sector
        )

        # 按因子类型过滤
        if factor_types:
            applicable_factors = [
                f for f in applicable_factors
                if f.get('factor_type') in factor_types
            ]

        for factor_config in applicable_factors:
            name = factor_config.get('name')
            factor_type_str = factor_config.get('factor_type', 'time_series')

            try:
                output = self._compute_single_factor(
                    name=name,
                    factor_type=factor_type_str,
                    data=data,
                    date=date,
                    asset=asset,
                    sector=sector,
                    params=factor_config.get('params', {}),
                    normalize=normalize
                )

                if output is not None:
                    outputs[name] = output

            except Exception as e:
                print(f"[FactorComputeEngine] Error computing {name}: {e}")
                continue

        return outputs

    def _compute_single_factor(
        self,
        name: str,
        factor_type: str,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        asset: str,
        sector: str,
        params: Dict,
        normalize: bool
    ) -> Optional[FactorOutput]:
        """计算单个因子"""
        # 获取因子实例
        factor = self.registry.get_or_create(name, **params)

        if factor is None:
            return None

        # 根据因子类型调用计算
        ft = factor.factor_type

        if ft == FactorType.TIME_SERIES:
            # 时序因子：单资产计算
            if asset and asset in data:
                raw_value = factor.calculate(data[asset])
            else:
                # 对所有资产计算
                raw_value = {}
                for ticker, df in data.items():
                    val = factor.calculate(df)
                    if val is not None:
                        raw_value[ticker] = val

        elif ft == FactorType.XS_GLOBAL:
            # 全局截面因子：全市场计算
            raw_value = factor.calculate(data, date=date)

        elif ft == FactorType.XS_PAIRWISE:
            # 配对因子：需要配对数据
            raw_value = factor.calculate(data, date=date)

        else:
            return None

        # 正则化
        if normalize:
            norm_method = getattr(factor, 'default_normalization', 'zscore_clip')
            values = self.normalizer.normalize(raw_value, method=norm_method)
            normalized = True
        else:
            values = raw_value
            normalized = False

        return FactorOutput(
            name=name,
            factor_type=ft,
            values=values,
            raw_values=raw_value,
            normalized=normalized,
            metadata={
                'factor_type': factor_type,
                'params': params
            }
        )

    def compute_for_asset(
        self,
        asset_data: pd.DataFrame,
        asset: str,
        sector: str = None,
        date: pd.Timestamp = None,
        normalize: bool = True
    ) -> Dict[str, FactorOutput]:
        """
        为单个资产计算所有时序因子

        Args:
            asset_data: 资产历史数据 DataFrame
            asset: 资产代码
            sector: 行业代码
            date: 当前日期
            normalize: 是否正则化

        Returns:
            {factor_name: FactorOutput}
        """
        outputs = {}

        # 获取时序因子
        ts_factors = self.param_resolver.get_applicable_factors(
            factor_type='time_series',
            asset=asset,
            sector=sector
        )

        for fc in ts_factors:
            name = fc.get('name')
            params = fc.get('params', {})

            factor = self.registry.get_or_create(name, **params)
            if factor is None:
                continue

            try:
                raw_value = factor.calculate(asset_data)

                if normalize:
                    norm_method = getattr(factor, 'default_normalization', 'zscore_clip')
                    values = self.normalizer.normalize(raw_value, method=norm_method)
                else:
                    values = raw_value

                outputs[name] = FactorOutput(
                    name=name,
                    factor_type=FactorType.TIME_SERIES,
                    values=values,
                    raw_values=raw_value,
                    normalized=normalize,
                    metadata={'asset': asset, 'sector': sector}
                )

            except Exception as e:
                print(f"[FactorComputeEngine] Error computing {name} for {asset}: {e}")
                continue

        return outputs

    def compute_for_pair(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        pair: Tuple[str, str],
        sector: str = None,
        date: pd.Timestamp = None,
        normalize: bool = True
    ) -> Dict[str, FactorOutput]:
        """
        为配对计算所有配对因子

        Args:
            data1, data2: 两个资产的数据
            pair: (asset1, asset2)
            sector: 行业代码
            date: 当前日期
            normalize: 是否正则化

        Returns:
            {factor_name: FactorOutput}
        """
        outputs = {}

        # 获取配对因子
        pair_factors = self.param_resolver.get_applicable_factors(
            factor_type='pair_trading',
            asset=pair[0],  # 使用第一个资产获取 idiosyncratic 配置
            sector=sector
        )

        for fc in pair_factors:
            name = fc.get('name')
            params = fc.get('params', {})

            factor = self.registry.get_or_create(name, **params)
            if factor is None:
                continue

            try:
                raw_value = factor.calculate(data1, data2)

                if normalize:
                    norm_method = getattr(factor, 'default_normalization', 'none')
                    values = self.normalizer.normalize(raw_value, method=norm_method)
                else:
                    values = raw_value

                outputs[name] = FactorOutput(
                    name=name,
                    factor_type=FactorType.XS_PAIRWISE,
                    values=values,
                    raw_values=raw_value,
                    normalized=normalize,
                    metadata={'pair': pair, 'sector': sector}
                )

            except Exception as e:
                print(f"[FactorComputeEngine] Error computing {name} for {pair}: {e}")
                continue

        return outputs

    def combine_signals(
        self,
        factor_outputs: Dict[str, FactorOutput],
        asset: str = None
    ) -> CombinedSignal:
        """
        组合因子信号

        Args:
            factor_outputs: {factor_name: FactorOutput}
            asset: 资产代码

        Returns:
            CombinedSignal: 组合后的信号
        """
        # 提取值和元数据
        values = {}
        metadata = {}

        for name, output in factor_outputs.items():
            if isinstance(output.values, (int, float)):
                values[name] = output.values
            elif isinstance(output.values, dict) and asset:
                values[name] = output.values.get(asset, 0.0)
            elif isinstance(output.values, pd.Series) and asset:
                values[name] = output.values.get(asset, 0.0)
            else:
                # 默认取第一个值或均值
                if isinstance(output.values, dict):
                    vals = list(output.values.values())
                    values[name] = vals[0] if vals else 0.0
                elif isinstance(output.values, pd.Series):
                    values[name] = output.values.iloc[0] if len(output.values) > 0 else 0.0

            metadata[name] = {
                'type': output.factor_type.name.lower() if output.factor_type else 'time_series',
                'category': output.metadata.get('category', 'unknown'),
                'scope': output.metadata.get('scope', 'common')
            }

        return self.combiner.combine(
            factor_outputs=values,
            asset=asset,
            factor_metadata=metadata
        )

    def __repr__(self) -> str:
        return f"FactorComputeEngine(config={self.config}, factors={len(self.registry)})"


# 便捷函数
def create_factor_engine(config_path: str = None) -> FactorComputeEngine:
    """创建因子计算引擎"""
    config = HierarchicalConfigLoader(config_path)
    config.load()
    return FactorComputeEngine(config=config)
