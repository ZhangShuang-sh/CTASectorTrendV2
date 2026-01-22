#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Factor Registry

Provides dynamic factor instantiation and management:
- FactorRegistry: Singleton registry for all factors
- Dynamic class loading from module path
- Factor caching for performance
- Metadata management
"""

from typing import Any, Dict, List, Optional, Type, Union
import importlib
from pathlib import Path

from core.factors.base import FactorBase, FactorType


class FactorRegistry:
    """
    因子注册表

    功能:
    1. 动态注册和实例化因子
    2. 通过名称或类路径获取因子
    3. 缓存因子实例以提高性能
    4. 管理因子元数据

    使用方式:
        registry = FactorRegistry()
        registry.register('HurstExponent', HurstExponent)
        factor = registry.get_or_create('HurstExponent', window=100)
    """

    _instance: Optional['FactorRegistry'] = None
    _initialized: bool = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 因子类注册表: {name: class}
        self._classes: Dict[str, Type[FactorBase]] = {}

        # 因子实例缓存: {(name, params_hash): instance}
        self._instances: Dict[tuple, FactorBase] = {}

        # 因子元数据: {name: metadata}
        self._metadata: Dict[str, Dict] = {}

        # 类路径映射: {name: module_path}
        self._class_paths: Dict[str, str] = {}

        self._initialized = True

    def register(
        self,
        name: str,
        factor_class: Type[FactorBase],
        metadata: Dict = None,
        class_path: str = None
    ) -> None:
        """
        注册因子类

        Args:
            name: 因子名称
            factor_class: 因子类
            metadata: 元数据 (可选)
            class_path: 类路径 (可选，用于延迟加载)
        """
        self._classes[name] = factor_class
        self._metadata[name] = metadata or {}

        if class_path:
            self._class_paths[name] = class_path

    def register_from_path(self, name: str, class_path: str, metadata: Dict = None) -> None:
        """
        通过类路径注册因子 (延迟加载)

        Args:
            name: 因子名称
            class_path: 类路径 (如 'core.factors.time_series.trend.HurstExponent')
            metadata: 元数据
        """
        self._class_paths[name] = class_path
        self._metadata[name] = metadata or {}

    def get_class(self, name: str) -> Optional[Type[FactorBase]]:
        """
        获取因子类

        Args:
            name: 因子名称

        Returns:
            因子类或 None
        """
        # 先检查已注册的类
        if name in self._classes:
            return self._classes[name]

        # 尝试从路径动态加载
        if name in self._class_paths:
            factor_class = self._load_class(self._class_paths[name])
            if factor_class:
                self._classes[name] = factor_class
                return factor_class

        return None

    def _load_class(self, class_path: str) -> Optional[Type[FactorBase]]:
        """
        从模块路径动态加载类

        Args:
            class_path: 完整类路径 (如 'core.factors.time_series.trend.HurstExponent')

        Returns:
            类对象或 None
        """
        try:
            parts = class_path.rsplit('.', 1)
            if len(parts) != 2:
                return None

            module_path, class_name = parts
            module = importlib.import_module(module_path)
            return getattr(module, class_name, None)

        except (ImportError, AttributeError) as e:
            print(f"[FactorRegistry] Failed to load class '{class_path}': {e}")
            return None

    def get_or_create(
        self,
        name: str,
        use_cache: bool = True,
        **kwargs
    ) -> Optional[FactorBase]:
        """
        获取或创建因子实例

        Args:
            name: 因子名称
            use_cache: 是否使用缓存
            **kwargs: 因子参数

        Returns:
            因子实例或 None
        """
        # 计算缓存键
        if use_cache:
            cache_key = self._make_cache_key(name, kwargs)
            if cache_key in self._instances:
                return self._instances[cache_key]

        # 获取类
        factor_class = self.get_class(name)
        if factor_class is None:
            print(f"[FactorRegistry] Factor '{name}' not found")
            return None

        # 创建实例
        try:
            # 获取默认参数
            default_params = self._metadata.get(name, {}).get('default_params', {})
            merged_params = {**default_params, **kwargs}

            instance = factor_class(name=name, **merged_params)

            # 缓存实例
            if use_cache:
                cache_key = self._make_cache_key(name, merged_params)
                self._instances[cache_key] = instance

            return instance

        except Exception as e:
            print(f"[FactorRegistry] Failed to create factor '{name}': {e}")
            return None

    def _make_cache_key(self, name: str, params: Dict) -> tuple:
        """生成缓存键"""
        # 将参数转换为可哈希的元组
        sorted_items = sorted(params.items())
        params_tuple = tuple((k, str(v)) for k, v in sorted_items)
        return (name, params_tuple)

    def get_metadata(self, name: str) -> Dict:
        """获取因子元数据"""
        return self._metadata.get(name, {}).copy()

    def list_factors(self, factor_type: FactorType = None) -> List[str]:
        """
        列出所有注册的因子

        Args:
            factor_type: 可选过滤类型

        Returns:
            因子名称列表
        """
        all_names = list(set(list(self._classes.keys()) + list(self._class_paths.keys())))

        if factor_type is None:
            return sorted(all_names)

        # 过滤类型
        result = []
        for name in all_names:
            factor_class = self.get_class(name)
            if factor_class and hasattr(factor_class, 'factor_type'):
                if factor_class.factor_type == factor_type:
                    result.append(name)

        return sorted(result)

    def clear_cache(self) -> None:
        """清空实例缓存"""
        self._instances.clear()

    def reset(self) -> None:
        """重置注册表"""
        self._classes.clear()
        self._instances.clear()
        self._metadata.clear()
        self._class_paths.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._classes or name in self._class_paths

    def __len__(self) -> int:
        return len(set(list(self._classes.keys()) + list(self._class_paths.keys())))

    def __repr__(self) -> str:
        return f"FactorRegistry(factors={len(self)}, cached={len(self._instances)})"


# 全局注册表实例
_global_registry: Optional[FactorRegistry] = None


def get_registry() -> FactorRegistry:
    """获取全局因子注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = FactorRegistry()
    return _global_registry


def register_factor(
    name: str,
    factor_class: Type[FactorBase] = None,
    class_path: str = None,
    metadata: Dict = None
) -> None:
    """
    便捷函数：注册因子到全局注册表

    Args:
        name: 因子名称
        factor_class: 因子类 (与 class_path 二选一)
        class_path: 类路径
        metadata: 元数据
    """
    registry = get_registry()

    if factor_class is not None:
        registry.register(name, factor_class, metadata, class_path)
    elif class_path is not None:
        registry.register_from_path(name, class_path, metadata)
    else:
        raise ValueError("Must provide either factor_class or class_path")


def get_factor(name: str, **kwargs) -> Optional[FactorBase]:
    """
    便捷函数：从全局注册表获取因子

    Args:
        name: 因子名称
        **kwargs: 因子参数

    Returns:
        因子实例
    """
    return get_registry().get_or_create(name, **kwargs)


# 装饰器：自动注册因子
def register(name: str = None, metadata: Dict = None):
    """
    装饰器：自动注册因子类

    Usage:
        @register('HurstExponent')
        class HurstExponent(TimeSeriesFactorBase):
            ...

        @register()  # 使用类名作为注册名
        class MyFactor(TimeSeriesFactorBase):
            ...
    """
    def decorator(cls: Type[FactorBase]):
        factor_name = name or cls.__name__
        get_registry().register(factor_name, cls, metadata)
        return cls
    return decorator
