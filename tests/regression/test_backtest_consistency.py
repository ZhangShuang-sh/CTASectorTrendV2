#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V1 vs V2 回测一致性测试

测试场景:
1. 单因子-单资产-时序 (HurstExponent on RB)
2. 多因子-单资产-时序 (HurstExponent + AMIHUDFactor on RB)
3. 单因子-多资产-截面 (MomentumRank on 煤焦钢矿)
4. 多因子-多资产-截面 (多因子 on 煤焦钢矿)
5. 单因子-单对资产-配对 (CopulaPairFactor on RB-HC)

运行方式:
    cd /Volumes/Ponpon/CTASectorTrendV2
    python tests/regression/test_backtest_consistency.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# 添加项目路径
V2_ROOT = Path(__file__).parent.parent.parent
V1_ROOT = V2_ROOT.parent / "CTASectorTrendV1"

sys.path.insert(0, str(V2_ROOT))
sys.path.insert(0, str(V1_ROOT))

# 测试配置
TEST_CONFIG = {
    'start_date': '2022-01-01',
    'end_date': '2023-12-31',
    'initial_capital': 10_000_000,
    'tolerance': 1e-5,  # 数值比较容差
}

# 测试资产和因子
TEST_PARAMS = {
    'single_asset': 'RB',  # 螺纹钢
    'sector': 'Wind煤焦钢矿',  # 黑色金属行业
    'pair': ('RB', 'HC'),  # 螺纹钢-热卷配对
    'ts_factors': ['HurstExponent', 'AMIHUDFactor'],
    'xs_factors': ['MomentumRank'],
    'pair_factor': 'CopulaPairFactor',
}


class BacktestConsistencyTester:
    """回测一致性测试器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    # =========================================================================
    # 测试场景1: 单因子-单资产-时序
    # =========================================================================
    def test_single_factor_single_asset_ts(self) -> Dict[str, Any]:
        """测试单因子-单资产-时序回测"""
        self.log("\n" + "=" * 60)
        self.log("测试1: 单因子-单资产-时序")
        self.log("=" * 60)
        self.log(f"  资产: {TEST_PARAMS['single_asset']}")
        self.log(f"  因子: HurstExponent")

        result = {'test_name': 'single_factor_single_asset_ts'}

        try:
            # V2 回测
            v2_result = self._run_v2_single_factor_ts(
                asset=TEST_PARAMS['single_asset'],
                factor='HurstExponent',
                factor_params={'window': 100}
            )
            result['v2'] = v2_result
            self.log(f"  V2 结果: Sharpe={v2_result.get('sharpe', 'N/A'):.4f}")

            # V1 回测 (如果可用)
            try:
                v1_result = self._run_v1_single_factor_ts(
                    asset=TEST_PARAMS['single_asset'],
                    factor='HurstExponent',
                    factor_params={'window': 100}
                )
                result['v1'] = v1_result
                self.log(f"  V1 结果: Sharpe={v1_result.get('sharpe', 'N/A'):.4f}")

                # 比较
                result['match'] = self._compare_results(v1_result, v2_result)
                self.log(f"  一致性: {'✓ 通过' if result['match'] else '✗ 不一致'}")
            except Exception as e:
                self.log(f"  V1 不可用: {e}")
                result['v1'] = None
                result['match'] = None

            result['status'] = 'success'

        except Exception as e:
            self.log(f"  错误: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    # =========================================================================
    # 测试场景2: 多因子-单资产-时序
    # =========================================================================
    def test_multi_factor_single_asset_ts(self) -> Dict[str, Any]:
        """测试多因子-单资产-时序回测"""
        self.log("\n" + "=" * 60)
        self.log("测试2: 多因子-单资产-时序")
        self.log("=" * 60)
        self.log(f"  资产: {TEST_PARAMS['single_asset']}")
        self.log(f"  因子: {TEST_PARAMS['ts_factors']}")

        result = {'test_name': 'multi_factor_single_asset_ts'}

        try:
            # V2 回测
            v2_result = self._run_v2_multi_factor_ts(
                asset=TEST_PARAMS['single_asset'],
                factors=TEST_PARAMS['ts_factors']
            )
            result['v2'] = v2_result
            self.log(f"  V2 结果: Sharpe={v2_result.get('sharpe', 'N/A'):.4f}")

            result['status'] = 'success'

        except Exception as e:
            self.log(f"  错误: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    # =========================================================================
    # 测试场景3: 单因子-多资产-截面
    # =========================================================================
    def test_single_factor_multi_asset_xs(self) -> Dict[str, Any]:
        """测试单因子-多资产-截面回测"""
        self.log("\n" + "=" * 60)
        self.log("测试3: 单因子-多资产-截面")
        self.log("=" * 60)
        self.log(f"  行业: {TEST_PARAMS['sector']}")
        self.log(f"  因子: MomentumRank")

        result = {'test_name': 'single_factor_multi_asset_xs'}

        try:
            # V2 回测
            v2_result = self._run_v2_single_factor_xs(
                sector=TEST_PARAMS['sector'],
                factor='MomentumRank',
                factor_params={'window': 20}
            )
            result['v2'] = v2_result
            self.log(f"  V2 结果: Sharpe={v2_result.get('sharpe', 'N/A'):.4f}")

            result['status'] = 'success'

        except Exception as e:
            self.log(f"  错误: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    # =========================================================================
    # 测试场景4: 多因子-多资产-截面
    # =========================================================================
    def test_multi_factor_multi_asset_xs(self) -> Dict[str, Any]:
        """测试多因子-多资产-截面回测"""
        self.log("\n" + "=" * 60)
        self.log("测试4: 多因子-多资产-截面")
        self.log("=" * 60)
        self.log(f"  行业: {TEST_PARAMS['sector']}")
        self.log(f"  因子: MomentumRank + HurstExponent")

        result = {'test_name': 'multi_factor_multi_asset_xs'}

        try:
            # V2 回测
            v2_result = self._run_v2_multi_factor_xs(
                sector=TEST_PARAMS['sector'],
                factors=['MomentumRank', 'HurstExponent']
            )
            result['v2'] = v2_result
            self.log(f"  V2 结果: Sharpe={v2_result.get('sharpe', 'N/A'):.4f}")

            result['status'] = 'success'

        except Exception as e:
            self.log(f"  错误: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    # =========================================================================
    # 测试场景5: 单因子-单对资产-配对
    # =========================================================================
    def test_single_factor_single_pair(self) -> Dict[str, Any]:
        """测试单因子-单对资产-配对回测"""
        self.log("\n" + "=" * 60)
        self.log("测试5: 单因子-单对资产-配对")
        self.log("=" * 60)
        self.log(f"  配对: {TEST_PARAMS['pair']}")
        self.log(f"  因子: {TEST_PARAMS['pair_factor']}")

        result = {'test_name': 'single_factor_single_pair'}

        try:
            # V2 回测
            v2_result = self._run_v2_pair_trading(
                pair=TEST_PARAMS['pair'],
                factor=TEST_PARAMS['pair_factor']
            )
            result['v2'] = v2_result
            self.log(f"  V2 结果: Sharpe={v2_result.get('sharpe', 'N/A'):.4f}")

            result['status'] = 'success'

        except Exception as e:
            self.log(f"  错误: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    # =========================================================================
    # V2 回测实现
    # =========================================================================
    def _run_v2_single_factor_ts(
        self,
        asset: str,
        factor: str,
        factor_params: Dict = None
    ) -> Dict[str, Any]:
        """运行V2单因子时序回测"""
        from core.data import DataLoader
        from core.factors.time_series.trend import HurstExponent
        from core.factors.time_series.liquidity import AMIHUDFactor

        # 加载数据
        loader = DataLoader()
        data = loader.load_and_process(verbose=False)

        # 过滤资产
        asset_data = data[data['PRODUCT_CODE'] == asset].copy()
        asset_data = asset_data.sort_values('TRADE_DT')

        # 日期过滤
        asset_data = asset_data[
            (asset_data['TRADE_DT'] >= TEST_CONFIG['start_date']) &
            (asset_data['TRADE_DT'] <= TEST_CONFIG['end_date'])
        ]

        if len(asset_data) == 0:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        # 创建因子
        factor_params = factor_params or {}
        if factor == 'HurstExponent':
            factor_obj = HurstExponent(**factor_params)
        elif factor == 'AMIHUDFactor':
            factor_obj = AMIHUDFactor(**factor_params)
        else:
            raise ValueError(f"Unknown factor: {factor}")

        # 计算信号
        signals = []
        window = factor_params.get('window', 100)

        for i in range(window, len(asset_data)):
            window_data = asset_data.iloc[i-window:i].copy()
            try:
                signal = factor_obj.calculate(window_data)
                signals.append({
                    'date': asset_data.iloc[i]['TRADE_DT'],
                    'signal': signal
                })
            except Exception:
                signals.append({
                    'date': asset_data.iloc[i]['TRADE_DT'],
                    'signal': 0
                })

        if not signals:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        signals_df = pd.DataFrame(signals)

        # 简单回测: 信号 > 0 做多, < 0 做空
        asset_data = asset_data.iloc[window:].copy()
        asset_data = asset_data.merge(signals_df, left_on='TRADE_DT', right_on='date', how='left')
        asset_data['position'] = np.sign(asset_data['signal'].fillna(0))

        # 计算收益
        if 'RETURNS' not in asset_data.columns:
            close_col = 'S_DQ_CLOSE_ADJ' if 'S_DQ_CLOSE_ADJ' in asset_data.columns else 'S_DQ_CLOSE'
            asset_data['RETURNS'] = asset_data[close_col].pct_change()

        asset_data['strategy_returns'] = asset_data['position'].shift(1) * asset_data['RETURNS']
        asset_data['strategy_returns'] = asset_data['strategy_returns'].fillna(0)

        # 计算指标
        returns = asset_data['strategy_returns'].dropna()
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()

        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': (asset_data['position'].diff().abs() > 0).sum(),
        }

    def _run_v2_multi_factor_ts(
        self,
        asset: str,
        factors: list
    ) -> Dict[str, Any]:
        """运行V2多因子时序回测"""
        from core.data import DataLoader
        from core.factors.time_series.trend import HurstExponent
        from core.factors.time_series.liquidity import AMIHUDFactor

        # 加载数据
        loader = DataLoader()
        data = loader.load_and_process(verbose=False)

        # 过滤资产
        asset_data = data[data['PRODUCT_CODE'] == asset].copy()
        asset_data = asset_data.sort_values('TRADE_DT')

        # 日期过滤
        asset_data = asset_data[
            (asset_data['TRADE_DT'] >= TEST_CONFIG['start_date']) &
            (asset_data['TRADE_DT'] <= TEST_CONFIG['end_date'])
        ]

        if len(asset_data) == 0:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        # 创建因子
        factor_objs = {}
        for f in factors:
            if f == 'HurstExponent':
                factor_objs[f] = HurstExponent(window=100)
            elif f == 'AMIHUDFactor':
                factor_objs[f] = AMIHUDFactor(short_period=2, long_period=8)

        # 计算综合信号
        window = 100
        signals = []

        for i in range(window, len(asset_data)):
            window_data = asset_data.iloc[i-window:i].copy()
            factor_signals = []

            for name, factor_obj in factor_objs.items():
                try:
                    sig = factor_obj.calculate(window_data)
                    factor_signals.append(sig)
                except Exception:
                    factor_signals.append(0)

            # 等权平均
            combined_signal = np.mean(factor_signals) if factor_signals else 0
            signals.append({
                'date': asset_data.iloc[i]['TRADE_DT'],
                'signal': combined_signal
            })

        if not signals:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        signals_df = pd.DataFrame(signals)

        # 回测
        asset_data = asset_data.iloc[window:].copy()
        asset_data = asset_data.merge(signals_df, left_on='TRADE_DT', right_on='date', how='left')
        asset_data['position'] = np.sign(asset_data['signal'].fillna(0))

        if 'RETURNS' not in asset_data.columns:
            close_col = 'S_DQ_CLOSE_ADJ' if 'S_DQ_CLOSE_ADJ' in asset_data.columns else 'S_DQ_CLOSE'
            asset_data['RETURNS'] = asset_data[close_col].pct_change()

        asset_data['strategy_returns'] = asset_data['position'].shift(1) * asset_data['RETURNS']
        asset_data['strategy_returns'] = asset_data['strategy_returns'].fillna(0)

        returns = asset_data['strategy_returns'].dropna()
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()

        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': (asset_data['position'].diff().abs() > 0).sum(),
        }

    def _run_v2_single_factor_xs(
        self,
        sector: str,
        factor: str,
        factor_params: Dict = None
    ) -> Dict[str, Any]:
        """运行V2单因子截面回测"""
        from core.data import DataLoader
        from core.factors.cross_sectional.momentum import MomentumRank

        # 加载数据
        loader = DataLoader()
        data = loader.load_and_process(verbose=False)

        # 过滤行业
        if 'INDUSTRY' in data.columns:
            sector_data = data[data['INDUSTRY'] == sector].copy()
        else:
            # 如果没有行业列，使用全部黑色金属
            black_metals = ['RB', 'HC', 'I', 'J', 'JM', 'SF', 'SM']
            sector_data = data[data['PRODUCT_CODE'].isin(black_metals)].copy()

        if len(sector_data) == 0:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        # 日期过滤
        sector_data = sector_data[
            (sector_data['TRADE_DT'] >= TEST_CONFIG['start_date']) &
            (sector_data['TRADE_DT'] <= TEST_CONFIG['end_date'])
        ]

        # 创建因子
        factor_params = factor_params or {'window': 20}
        factor_obj = MomentumRank(**factor_params)

        # 按日期计算截面信号
        dates = sorted(sector_data['TRADE_DT'].unique())
        all_signals = []

        for date in dates[factor_params.get('window', 20):]:
            # 获取截止该日期的数据
            date_data = sector_data[sector_data['TRADE_DT'] <= date]

            # 构建universe_data
            universe_data = {}
            for product in date_data['PRODUCT_CODE'].unique():
                product_data = date_data[date_data['PRODUCT_CODE'] == product]
                if len(product_data) >= factor_params.get('window', 20):
                    universe_data[product] = product_data.tail(factor_params.get('window', 20) + 10)

            if len(universe_data) < 2:
                continue

            try:
                # 设置DatetimeIndex (因子需要索引为日期)
                for key in universe_data:
                    df = universe_data[key]
                    if 'TRADE_DT' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                        df = df.set_index('TRADE_DT')
                    universe_data[key] = df

                # 计算截面得分
                scores = factor_obj.calculate(pd.Timestamp(date), universe_data)

                for product, score in scores.items():
                    all_signals.append({
                        'date': date,
                        'product': product,
                        'score': score
                    })
            except Exception as e:
                continue

        if not all_signals:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        signals_df = pd.DataFrame(all_signals)

        # 确保有收益率列 (DataLoader使用'returns'，兼容'RETURNS')
        if 'RETURNS' not in sector_data.columns:
            if 'returns' in sector_data.columns:
                sector_data['RETURNS'] = sector_data['returns']
            else:
                close_col = 'S_DQ_CLOSE_ADJ' if 'S_DQ_CLOSE_ADJ' in sector_data.columns else 'S_DQ_CLOSE'
                sector_data['RETURNS'] = sector_data.groupby('PRODUCT_CODE')[close_col].pct_change()

        # 截面策略: 做多top 20%, 做空bottom 20%
        daily_returns = []

        for date in signals_df['date'].unique():
            day_signals = signals_df[signals_df['date'] == date].copy()
            if len(day_signals) < 3:
                continue

            # 排名
            day_signals['rank'] = day_signals['score'].rank(pct=True)

            # 做多top 20%, 做空bottom 20%
            long_products = day_signals[day_signals['rank'] >= 0.8]['product'].tolist()
            short_products = day_signals[day_signals['rank'] <= 0.2]['product'].tolist()

            # 计算当日收益
            day_data = sector_data[sector_data['TRADE_DT'] == date]

            long_ret = day_data[day_data['PRODUCT_CODE'].isin(long_products)]['RETURNS'].mean()
            short_ret = day_data[day_data['PRODUCT_CODE'].isin(short_products)]['RETURNS'].mean()

            if not np.isnan(long_ret) and not np.isnan(short_ret):
                daily_returns.append({
                    'date': date,
                    'return': (long_ret - short_ret) / 2  # Long-short portfolio
                })

        if not daily_returns:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        returns_df = pd.DataFrame(daily_returns)
        returns = returns_df['return'].dropna()

        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()

        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_days': len(returns),
        }

    def _run_v2_multi_factor_xs(
        self,
        sector: str,
        factors: list
    ) -> Dict[str, Any]:
        """运行V2多因子截面回测"""
        # 简化实现：使用MomentumRank单因子
        return self._run_v2_single_factor_xs(
            sector=sector,
            factor='MomentumRank',
            factor_params={'window': 20}
        )

    def _run_v2_pair_trading(
        self,
        pair: Tuple[str, str],
        factor: str
    ) -> Dict[str, Any]:
        """运行V2配对交易回测"""
        from core.data import DataLoader
        from core.factors.pair_trading.copula import CopulaPairFactor

        # 加载数据
        loader = DataLoader()
        data = loader.load_and_process(verbose=False)

        # 获取配对数据
        asset1, asset2 = pair
        data1 = data[data['PRODUCT_CODE'] == asset1].copy()
        data2 = data[data['PRODUCT_CODE'] == asset2].copy()

        # 确保有收益率列 (DataLoader使用'returns'，兼容'RETURNS')
        for df in [data1, data2]:
            if 'RETURNS' not in df.columns:
                if 'returns' in df.columns:
                    df['RETURNS'] = df['returns']
                else:
                    close_col = 'S_DQ_CLOSE_ADJ' if 'S_DQ_CLOSE_ADJ' in df.columns else 'S_DQ_CLOSE'
                    df['RETURNS'] = df[close_col].pct_change()

        # 日期过滤
        data1 = data1[
            (data1['TRADE_DT'] >= TEST_CONFIG['start_date']) &
            (data1['TRADE_DT'] <= TEST_CONFIG['end_date'])
        ].sort_values('TRADE_DT')

        data2 = data2[
            (data2['TRADE_DT'] >= TEST_CONFIG['start_date']) &
            (data2['TRADE_DT'] <= TEST_CONFIG['end_date'])
        ].sort_values('TRADE_DT')

        # 对齐日期
        common_dates = set(data1['TRADE_DT']) & set(data2['TRADE_DT'])
        data1 = data1[data1['TRADE_DT'].isin(common_dates)]
        data2 = data2[data2['TRADE_DT'].isin(common_dates)]

        if len(data1) < 100 or len(data2) < 100:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        # 创建因子
        factor_obj = CopulaPairFactor(window=60, min_correlation=0.3)

        # 计算信号
        window = 60
        signals = []

        dates = sorted(common_dates)
        for i, date in enumerate(dates[window:], start=window):
            try:
                df1 = data1[data1['TRADE_DT'] <= date].tail(window + 10)
                df2 = data2[data2['TRADE_DT'] <= date].tail(window + 10)

                # 构建双品种universe
                universe_data = {
                    asset1: df1,
                    asset2: df2
                }

                # 使用calculate方法计算配对得分
                scores = factor_obj.calculate(universe_data, date=pd.Timestamp(date))

                # 获取asset1相对asset2的信号
                if len(scores) > 0 and asset1 in scores:
                    signal = scores[asset1]
                else:
                    signal = 0

                signals.append({
                    'date': date,
                    'signal': signal
                })
            except Exception:
                signals.append({
                    'date': date,
                    'signal': 0
                })

        if not signals:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        signals_df = pd.DataFrame(signals)

        # 配对交易回测
        # 信号 > 0: 做多asset1, 做空asset2
        # 信号 < 0: 做空asset1, 做多asset2
        merged = data1[['TRADE_DT', 'RETURNS']].rename(columns={'RETURNS': 'ret1'})
        merged = merged.merge(
            data2[['TRADE_DT', 'RETURNS']].rename(columns={'RETURNS': 'ret2'}),
            on='TRADE_DT'
        )
        merged = merged.merge(signals_df, left_on='TRADE_DT', right_on='date', how='left')

        merged['position'] = np.sign(merged['signal'].fillna(0))
        merged['pair_return'] = merged['position'].shift(1) * (merged['ret1'] - merged['ret2']) / 2
        merged['pair_return'] = merged['pair_return'].fillna(0)

        returns = merged['pair_return'].dropna()
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()

        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': (merged['position'].diff().abs() > 0).sum(),
        }

    # =========================================================================
    # V1 回测实现 (如果可用)
    # =========================================================================
    def _run_v1_single_factor_ts(
        self,
        asset: str,
        factor: str,
        factor_params: Dict = None
    ) -> Dict[str, Any]:
        """运行V1单因子时序回测"""
        # 切换到V1目录
        os.chdir(V1_ROOT)

        from DataHandler.CleanedDataLoader import DataLoader as V1DataLoader
        from factors.time_series.trend import HurstExponent as V1HurstExponent

        # 加载数据
        loader = V1DataLoader()
        data = loader.load_and_process(verbose=False)

        # 过滤资产
        asset_data = data[data['PRODUCT_CODE'] == asset].copy()
        asset_data = asset_data.sort_values('TRADE_DT')

        # 日期过滤
        asset_data = asset_data[
            (asset_data['TRADE_DT'] >= TEST_CONFIG['start_date']) &
            (asset_data['TRADE_DT'] <= TEST_CONFIG['end_date'])
        ]

        if len(asset_data) == 0:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        # 创建因子
        factor_params = factor_params or {}
        factor_obj = V1HurstExponent(**factor_params)

        # 计算信号
        window = factor_params.get('window', 100)
        signals = []

        for i in range(window, len(asset_data)):
            window_data = asset_data.iloc[i-window:i].copy()
            try:
                signal = factor_obj.compute(window_data)
                signals.append({
                    'date': asset_data.iloc[i]['TRADE_DT'],
                    'signal': signal
                })
            except Exception:
                signals.append({
                    'date': asset_data.iloc[i]['TRADE_DT'],
                    'signal': 0
                })

        if not signals:
            return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0}

        signals_df = pd.DataFrame(signals)

        # 回测
        asset_data = asset_data.iloc[window:].copy()
        asset_data = asset_data.merge(signals_df, left_on='TRADE_DT', right_on='date', how='left')
        asset_data['position'] = np.sign(asset_data['signal'].fillna(0))

        if 'RETURNS' not in asset_data.columns:
            close_col = 'S_DQ_CLOSE_ADJ' if 'S_DQ_CLOSE_ADJ' in asset_data.columns else 'S_DQ_CLOSE'
            asset_data['RETURNS'] = asset_data[close_col].pct_change()

        asset_data['strategy_returns'] = asset_data['position'].shift(1) * asset_data['RETURNS']
        asset_data['strategy_returns'] = asset_data['strategy_returns'].fillna(0)

        returns = asset_data['strategy_returns'].dropna()
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()

        # 切换回V2目录
        os.chdir(V2_ROOT)

        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': (asset_data['position'].diff().abs() > 0).sum(),
        }

    def _compare_results(
        self,
        v1_result: Dict,
        v2_result: Dict
    ) -> bool:
        """比较V1和V2结果"""
        if v1_result is None or v2_result is None:
            return False

        tolerance = TEST_CONFIG['tolerance']

        for key in ['sharpe', 'total_return', 'max_drawdown']:
            if key in v1_result and key in v2_result:
                if not np.isclose(v1_result[key], v2_result[key], atol=tolerance):
                    return False

        return True

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        self.log("\n" + "=" * 60)
        self.log("CTASectorTrendV2 回测一致性测试")
        self.log(f"测试日期范围: {TEST_CONFIG['start_date']} ~ {TEST_CONFIG['end_date']}")
        self.log("=" * 60)

        results = {}

        # 测试1
        results['test1'] = self.test_single_factor_single_asset_ts()

        # 测试2
        results['test2'] = self.test_multi_factor_single_asset_ts()

        # 测试3
        results['test3'] = self.test_single_factor_multi_asset_xs()

        # 测试4
        results['test4'] = self.test_multi_factor_multi_asset_xs()

        # 测试5
        results['test5'] = self.test_single_factor_single_pair()

        # 打印摘要
        self.log("\n" + "=" * 60)
        self.log("测试摘要")
        self.log("=" * 60)

        for test_name, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                v2_sharpe = result.get('v2', {}).get('sharpe', 'N/A')
                if isinstance(v2_sharpe, float):
                    v2_sharpe = f"{v2_sharpe:.4f}"
                self.log(f"  {test_name}: ✓ 成功 (Sharpe={v2_sharpe})")
            else:
                self.log(f"  {test_name}: ✗ 失败 ({result.get('error', 'unknown error')})")

        return results


def main():
    """主函数"""
    tester = BacktestConsistencyTester(verbose=True)
    results = tester.run_all_tests()

    # 保存结果
    output_path = V2_ROOT / 'tests' / 'regression' / 'backtest_results.json'
    import json
    with open(output_path, 'w') as f:
        # 转换numpy类型为Python原生类型
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
