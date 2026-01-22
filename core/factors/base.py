#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Unified Factor Base Classes

Provides the abstract base class for all factors with:
- Unified calculate() interface (replacing V1's compute())
- Factor type classification (TIME_SERIES, XS_PAIRWISE, XS_GLOBAL)
- Factor scope definition (COMMON, IDIOSYNCRATIC)
- Backward compatibility with V1's compute() method

Design Goals:
1. Factors are completely decoupled from execution engine
2. All factors implement the same calculate() interface
3. 100% logic consistency with V1 (np.allclose(atol=1e-7))
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from enum import Enum, auto
import pandas as pd


class FactorType(Enum):
    """
    因子计算范式枚举

    定义三种因子类型的计算方式:
    - TIME_SERIES: 时序因子，对单一资产独立计算信号
    - XS_PAIRWISE: 配对截面因子，比较特定的2个资产
    - XS_GLOBAL: 全局截面因子，在N个资产间进行排名
    """
    TIME_SERIES = auto()      # 单资产时序信号 → float in [-1, 1]
    XS_PAIRWISE = auto()      # 配对资产比较 (2个资产) → float in [-1, 1]
    XS_GLOBAL = auto()        # 全局截面排名 (N个资产) → pd.Series


class FactorScope(Enum):
    """
    因子应用范围枚举

    定义因子的应用范围:
    - COMMON: 通用因子，应用于所有资产
    - IDIOSYNCRATIC: 特定因子，应用于特定资产或资产类别
    """
    COMMON = auto()           # 通用因子
    IDIOSYNCRATIC = auto()    # 特定因子


class FactorBase(ABC):
    """
    所有量化因子的抽象基类 (V2版本)

    V2变更:
    - 核心方法从 compute() 更名为 calculate()
    - compute() 保留作为 calculate() 的别名，保持向后兼容
    - 新增 default_normalization 属性

    Attributes:
        name (str): 因子名称
        window (int): 回看窗口长度
        factor_type (FactorType): 因子类型 (子类必须声明)
        default_normalization (str): 默认正则化方法
        _params (Dict): 当前参数缓存
    """

    # 子类必须声明因子类型
    factor_type: FactorType = NotImplemented

    # 默认正则化方法 (zscore_clip / minmax / rank / none)
    default_normalization: str = "zscore_clip"

    def __init__(self, name: str, window: int = 20):
        """
        Args:
            name (str): 因子名称 (例如: 'Hurst_100')
            window (int): 回看窗口长度 (例如: 100)
        """
        self.name = name
        self.window = window
        self._params: Dict[str, Any] = {
            'name': name,
            'window': window
        }

    @abstractmethod
    def calculate(self, *args, **kwargs) -> Union[float, pd.Series, Dict]:
        """
        核心计算方法 - V2统一接口

        替代V1的compute()方法，所有因子必须实现此方法。

        Returns:
            - TIME_SERIES: float in [-1, 1]
            - XS_PAIRWISE: float in [-1, 1]
            - XS_GLOBAL: pd.Series with asset keys
        """
        pass

    def compute(self, *args, **kwargs) -> Any:
        """
        向后兼容方法 - 调用 calculate()

        V1代码中使用 compute() 的地方将自动调用 calculate()。
        """
        return self.calculate(*args, **kwargs)

    def set_params(self, **kwargs) -> 'FactorBase':
        """
        动态设置因子参数

        Args:
            **kwargs: 要更新的参数键值对

        Returns:
            FactorBase: 返回自身以支持链式调用
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            self._params[key] = value
        return self

    def get_params(self) -> Dict[str, Any]:
        """
        获取当前因子参数

        Returns:
            Dict[str, Any]: 参数字典的副本
        """
        return self._params.copy()

    @property
    def requires_pair(self) -> bool:
        """是否需要配对定义"""
        return self.factor_type == FactorType.XS_PAIRWISE

    @property
    def requires_universe(self) -> bool:
        """是否需要全市场数据"""
        return self.factor_type == FactorType.XS_GLOBAL

    def validate_inputs(self, *args, **kwargs) -> bool:
        """
        验证输入数据的有效性

        子类可以覆盖此方法添加特定验证逻辑。

        Returns:
            bool: 输入是否有效
        """
        return True

    def __repr__(self) -> str:
        params_str = ', '.join(f'{k}={v}' for k, v in self._params.items())
        return f"{self.__class__.__name__}({params_str})"


# Alias for backward compatibility with V1
BaseFactor = FactorBase
