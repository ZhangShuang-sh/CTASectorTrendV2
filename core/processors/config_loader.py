#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Hierarchical Configuration Loader

Supports 4-layer configuration hierarchy:
- L1: Factor Type (time_series, cross_sectional, pair_trading)
- L2: Application Scope (common, idiosyncratic)
- L3: Logical Category (trend, volatility, liquidity, etc.)
- L4: Sub-category (specific factor implementations)

Configuration priority: Asset > Sector > Common
"""

import os
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import copy


class HierarchicalConfigLoader:
    """
    层级配置加载器 (V2版本)

    支持四层配置层级:
    - L1: 因子类型 (factor_type_weights)
    - L2: 应用范围 (common / idiosyncratic)
    - L3: 逻辑大类 (trend, volatility, liquidity, etc.)
    - L4: 子类 (具体因子实现)

    参数解析优先级: 资产 > 行业 > 全局
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: 配置文件路径，默认为 config/multi_factor_config.yaml
        """
        if config_path is None:
            # 从 processors/ 目录相对定位到 config/
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "multi_factor_config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict = {}
        self._loaded = False

    def load(self) -> Dict:
        """
        加载配置文件

        Returns:
            Dict: 完整配置字典
        """
        if self._loaded:
            return self._config

        if not self.config_path.exists():
            # 返回默认配置
            self._config = self._get_default_config()
            self._loaded = True
            return self._config

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}

        self._loaded = True
        return self._config

    def reload(self) -> Dict:
        """强制重新加载配置"""
        self._loaded = False
        return self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的路径）

        Args:
            key: 配置键，支持点号分隔 (如 'factor_type_weights.time_series')
            default: 默认值

        Returns:
            配置值或默认值
        """
        self.load()
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    # ========== L1: Factor Type ==========

    def get_factor_type_weights(self) -> Dict[str, float]:
        """
        获取 L1 因子类型权重

        Returns:
            {factor_type: weight}
        """
        self.load()
        return copy.deepcopy(
            self._config.get('factor_type_weights', {
                'time_series': 0.50,
                'cross_sectional': 0.35,
                'pair_trading': 0.15
            })
        )

    # ========== L2: Application Scope ==========

    def get_common_factors(self, factor_type: str = None) -> List[Dict]:
        """
        获取 Common 因子配置

        Args:
            factor_type: 可选类型过滤 ('time_series', 'cross_sectional', 'pair_trading')

        Returns:
            因子配置列表
        """
        self.load()
        common = self._config.get('common', {})

        if factor_type is None:
            # 返回所有
            all_factors = []
            for ft in ['time_series', 'cross_sectional', 'pair_trading']:
                all_factors.extend(common.get(ft, []))
            return copy.deepcopy(all_factors)

        return copy.deepcopy(common.get(factor_type, []))

    def get_idiosyncratic_factors(
        self,
        factor_type: str = None,
        sector: str = None,
        asset: str = None
    ) -> List[Dict]:
        """
        获取 Idiosyncratic 因子配置

        Args:
            factor_type: 因子类型过滤
            sector: 行业过滤
            asset: 资产过滤

        Returns:
            因子配置列表
        """
        self.load()
        idio = self._config.get('idiosyncratic', {})

        factors = []

        # 行业级配置
        if sector:
            sector_config = idio.get('by_sector', {}).get(sector, {})
            if factor_type:
                factors.extend(sector_config.get(factor_type, []))
            else:
                for ft in ['time_series', 'cross_sectional', 'pair_trading']:
                    factors.extend(sector_config.get(ft, []))

        # 资产级配置
        if asset:
            asset_config = idio.get('by_asset', {}).get(asset, {})
            if factor_type:
                factors.extend(asset_config.get(factor_type, []))
            else:
                for ft in ['time_series', 'cross_sectional', 'pair_trading']:
                    factors.extend(asset_config.get(ft, []))

        return copy.deepcopy(factors)

    # ========== L3: Logical Category ==========

    def get_factors_by_category(
        self,
        category: str,
        factor_type: str = None
    ) -> List[Dict]:
        """
        按逻辑大类获取因子

        Args:
            category: 逻辑大类 ('trend', 'volatility', 'liquidity', 'momentum', etc.)
            factor_type: 因子类型过滤

        Returns:
            该大类下的所有因子配置
        """
        self.load()

        # 从所有因子中过滤
        all_factors = self.get_common_factors(factor_type)
        all_factors.extend(self.get_idiosyncratic_factors(factor_type=factor_type))

        return [f for f in all_factors if f.get('category') == category]

    def get_categories(self, factor_type: str = None) -> List[str]:
        """
        获取所有逻辑大类名称

        Args:
            factor_type: 因子类型过滤

        Returns:
            大类名称列表
        """
        all_factors = self.get_common_factors(factor_type)

        categories = set()
        for f in all_factors:
            if 'category' in f:
                categories.add(f['category'])

        return sorted(list(categories))

    # ========== L4: Sub-category ==========

    def get_factor_config(
        self,
        factor_name: str,
        factor_type: str = None
    ) -> Optional[Dict]:
        """
        获取指定因子的配置

        Args:
            factor_name: 因子名称
            factor_type: 因子类型 (可选)

        Returns:
            因子配置字典或 None
        """
        self.load()

        # 搜索 common
        common = self._config.get('common', {})
        for ft in (['time_series', 'cross_sectional', 'pair_trading']
                   if factor_type is None else [factor_type]):
            for fc in common.get(ft, []):
                if fc.get('name') == factor_name:
                    return copy.deepcopy(fc)

        return None

    def get_factor_default_params(self, factor_name: str) -> Dict:
        """
        获取因子的默认参数

        Args:
            factor_name: 因子名称

        Returns:
            默认参数字典
        """
        fc = self.get_factor_config(factor_name)
        if fc is None:
            return {}
        return copy.deepcopy(fc.get('default_params', {}))

    # ========== Hierarchical Parameter Resolution ==========

    def resolve_factor_params(
        self,
        factor_name: str,
        factor_type: str = None,
        asset: str = None,
        sector: str = None
    ) -> Dict:
        """
        解析因子参数（支持层级覆盖）

        优先级: Asset > Sector > Common

        Args:
            factor_name: 因子名称
            factor_type: 因子类型
            asset: 资产代码
            sector: 行业代码

        Returns:
            合并后的参数字典
        """
        self.load()

        # 1. 获取 Common 默认参数
        params = self.get_factor_default_params(factor_name)

        # 2. 应用 Sector 覆盖
        if sector:
            sector_overrides = self._get_override_params(
                factor_name, factor_type, 'by_sector', sector
            )
            params = self._merge_params(params, sector_overrides)

        # 3. 应用 Asset 覆盖
        if asset:
            asset_overrides = self._get_override_params(
                factor_name, factor_type, 'by_asset', asset
            )
            params = self._merge_params(params, asset_overrides)

        return params

    def _get_override_params(
        self,
        factor_name: str,
        factor_type: str,
        level: str,
        key: str
    ) -> Dict:
        """获取特定层级的覆盖参数"""
        idio = self._config.get('idiosyncratic', {})
        level_config = idio.get(level, {})
        entity_config = level_config.get(key, {})

        factor_types = [factor_type] if factor_type else \
                       ['time_series', 'cross_sectional', 'pair_trading']

        for ft in factor_types:
            for fc in entity_config.get(ft, []):
                if fc.get('name') == factor_name:
                    return fc.get('params', {})

        return {}

    def _merge_params(self, base: Dict, override: Dict) -> Dict:
        """合并参数，override 覆盖 base"""
        result = copy.deepcopy(base)
        result.update(override)
        return result

    # ========== Applicable Factors ==========

    def get_applicable_factors(
        self,
        factor_type: str = None,
        asset: str = None,
        sector: str = None
    ) -> List[Dict]:
        """
        获取所有适用的因子配置

        Common 因子 + Idiosyncratic 因子都必须计算！

        Args:
            factor_type: 因子类型过滤
            asset: 资产代码
            sector: 行业代码

        Returns:
            适用因子配置列表 (包含 Common + Idiosyncratic)
        """
        # 1. Common 因子 (始终适用)
        common_factors = self.get_common_factors(factor_type)

        # 2. Idiosyncratic 因子 (根据 asset/sector 过滤)
        idio_factors = self.get_idiosyncratic_factors(
            factor_type=factor_type,
            sector=sector,
            asset=asset
        )

        # 3. 合并 (去重)
        all_factors = []
        seen_names = set()

        for f in common_factors + idio_factors:
            name = f.get('name')
            if name and name not in seen_names:
                all_factors.append(f)
                seen_names.add(name)

        return all_factors

    # ========== Backtest Config ==========

    def get_backtest_config(self) -> Dict:
        """获取回测配置"""
        self.load()
        return copy.deepcopy(self._config.get('backtest', {}))

    def get_pairs_config(self) -> Dict[str, List[Tuple[str, str]]]:
        """获取配对配置"""
        self.load()
        fixed_pairs = self._config.get('fixed_pairs', {})
        return {
            industry: [tuple(pair) for pair in pairs]
            for industry, pairs in fixed_pairs.items()
        }

    # ========== Default Config ==========

    def _get_default_config(self) -> Dict:
        """返回默认配置"""
        return {
            'factor_type_weights': {
                'time_series': 0.50,
                'cross_sectional': 0.35,
                'pair_trading': 0.15
            },
            'common': {
                'time_series': [
                    {
                        'name': 'HurstExponent',
                        'class': 'core.factors.time_series.trend.HurstExponent',
                        'category': 'trend',
                        'default_params': {'window': 100, 'scale': 2.0}
                    },
                    {
                        'name': 'AMIHUDFactor',
                        'class': 'core.factors.time_series.liquidity.AMIHUDFactor',
                        'category': 'liquidity',
                        'default_params': {'short_period': 2, 'long_period': 8}
                    }
                ],
                'cross_sectional': [],
                'pair_trading': [
                    {
                        'name': 'CopulaPairFactor',
                        'class': 'core.factors.pair_trading.copula.CopulaPairFactor',
                        'category': 'copula',
                        'default_params': {'window': 60, 'min_correlation': 0.5}
                    }
                ]
            },
            'idiosyncratic': {
                'by_sector': {},
                'by_asset': {}
            },
            'backtest': {
                'initial_capital': 10000000,
                'transaction_cost': 0.001,
                'slippage': 0.0002,
                'margin_rate': 0.1
            }
        }

    def __repr__(self) -> str:
        return f"HierarchicalConfigLoader(path={self.config_path}, loaded={self._loaded})"


# 便捷函数
def load_config(config_path: str = None) -> HierarchicalConfigLoader:
    """
    便捷函数：加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        HierarchicalConfigLoader 实例
    """
    loader = HierarchicalConfigLoader(config_path)
    loader.load()
    return loader
