#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Parameter Resolver

Resolves factor parameters with hierarchical override:
- Priority: Asset > Sector > Common

Critical: Common + Idiosyncratic factors must BOTH be computed!
"""

from typing import Any, Dict, List, Optional, Tuple
import copy

from core.processors.config_loader import HierarchicalConfigLoader


class ParameterResolver:
    """
    参数解析器

    功能:
    1. 解析因子参数优先级: Asset > Sector > Common
    2. 返回所有适用因子 (Common + Idiosyncratic 都必须计算!)
    3. 支持动态参数覆盖
    """

    def __init__(self, config_loader: HierarchicalConfigLoader = None):
        """
        Args:
            config_loader: 配置加载器实例
        """
        if config_loader is None:
            config_loader = HierarchicalConfigLoader()
            config_loader.load()

        self.config = config_loader

    def resolve_params(
        self,
        factor_name: str,
        factor_type: str = None,
        asset: str = None,
        sector: str = None,
        **overrides
    ) -> Dict[str, Any]:
        """
        解析因子参数

        优先级: overrides > Asset > Sector > Common

        Args:
            factor_name: 因子名称
            factor_type: 因子类型 ('time_series', 'cross_sectional', 'pair_trading')
            asset: 资产代码 (如 'RB')
            sector: 行业代码 (如 'Wind煤焦钢矿')
            **overrides: 手动覆盖参数

        Returns:
            合并后的参数字典
        """
        # 1. 获取 Common 默认参数
        params = self._get_common_params(factor_name, factor_type)

        # 2. 应用 Sector 覆盖
        if sector:
            sector_params = self._get_sector_params(factor_name, factor_type, sector)
            params = self._merge(params, sector_params)

        # 3. 应用 Asset 覆盖
        if asset:
            asset_params = self._get_asset_params(factor_name, factor_type, asset)
            params = self._merge(params, asset_params)

        # 4. 应用手动覆盖
        if overrides:
            params = self._merge(params, overrides)

        return params

    def _get_common_params(self, factor_name: str, factor_type: str = None) -> Dict:
        """获取 Common 默认参数"""
        return self.config.get_factor_default_params(factor_name)

    def _get_sector_params(
        self,
        factor_name: str,
        factor_type: str,
        sector: str
    ) -> Dict:
        """获取 Sector 覆盖参数"""
        idio_factors = self.config.get_idiosyncratic_factors(
            factor_type=factor_type,
            sector=sector
        )

        for fc in idio_factors:
            if fc.get('name') == factor_name:
                return fc.get('params', {})

        return {}

    def _get_asset_params(
        self,
        factor_name: str,
        factor_type: str,
        asset: str
    ) -> Dict:
        """获取 Asset 覆盖参数"""
        idio_factors = self.config.get_idiosyncratic_factors(
            factor_type=factor_type,
            asset=asset
        )

        for fc in idio_factors:
            if fc.get('name') == factor_name:
                return fc.get('params', {})

        return {}

    def _merge(self, base: Dict, override: Dict) -> Dict:
        """合并参数，override 覆盖 base"""
        result = copy.deepcopy(base)
        result.update(override)
        return result

    def get_applicable_factors(
        self,
        factor_type: str = None,
        asset: str = None,
        sector: str = None
    ) -> List[Dict]:
        """
        获取所有适用因子

        CRITICAL: Common 因子 + Idiosyncratic 因子都必须计算!

        Args:
            factor_type: 因子类型过滤
            asset: 资产代码
            sector: 行业代码

        Returns:
            适用因子配置列表，每个因子包含:
            - name: 因子名称
            - class: 因子类路径
            - factor_type: 因子类型
            - category: 逻辑大类
            - params: 解析后的参数 (已应用层级覆盖)
            - scope: 'common' 或 'idiosyncratic'
        """
        result = []
        seen_names = set()

        # 1. Common 因子 (始终适用)
        common_factors = self.config.get_common_factors(factor_type)

        for fc in common_factors:
            name = fc.get('name')
            if name in seen_names:
                continue

            # 解析参数
            params = self.resolve_params(
                factor_name=name,
                factor_type=fc.get('factor_type', factor_type),
                asset=asset,
                sector=sector
            )

            result.append({
                'name': name,
                'class': fc.get('class'),
                'factor_type': fc.get('factor_type', factor_type),
                'category': fc.get('category'),
                'params': params,
                'scope': 'common'
            })
            seen_names.add(name)

        # 2. Idiosyncratic 因子 (根据 asset/sector)
        idio_factors = self.config.get_idiosyncratic_factors(
            factor_type=factor_type,
            sector=sector,
            asset=asset
        )

        for fc in idio_factors:
            name = fc.get('name')
            if name in seen_names:
                continue

            # Idiosyncratic 因子直接使用配置中的参数
            result.append({
                'name': name,
                'class': fc.get('class'),
                'factor_type': fc.get('factor_type', factor_type),
                'category': fc.get('category'),
                'params': fc.get('params', {}),
                'scope': 'idiosyncratic'
            })
            seen_names.add(name)

        return result

    def get_factors_by_category(
        self,
        category: str,
        factor_type: str = None,
        asset: str = None,
        sector: str = None
    ) -> List[Dict]:
        """
        按逻辑大类获取适用因子

        Args:
            category: 逻辑大类 ('trend', 'volatility', etc.)
            factor_type: 因子类型过滤
            asset: 资产代码
            sector: 行业代码

        Returns:
            该大类下的适用因子列表
        """
        all_factors = self.get_applicable_factors(
            factor_type=factor_type,
            asset=asset,
            sector=sector
        )

        return [f for f in all_factors if f.get('category') == category]

    def get_factor_type_allocation(
        self,
        asset: str = None,
        sector: str = None
    ) -> Dict[str, List[Dict]]:
        """
        按因子类型分组获取适用因子

        Args:
            asset: 资产代码
            sector: 行业代码

        Returns:
            {factor_type: [factor_configs]}
        """
        allocation = {
            'time_series': [],
            'cross_sectional': [],
            'pair_trading': []
        }

        for ft in allocation.keys():
            allocation[ft] = self.get_applicable_factors(
                factor_type=ft,
                asset=asset,
                sector=sector
            )

        return allocation

    def __repr__(self) -> str:
        return f"ParameterResolver(config={self.config})"


# 便捷函数
def create_param_resolver(config_path: str = None) -> ParameterResolver:
    """
    便捷函数：创建参数解析器

    Args:
        config_path: 配置文件路径

    Returns:
        ParameterResolver 实例
    """
    config = HierarchicalConfigLoader(config_path)
    config.load()
    return ParameterResolver(config)
