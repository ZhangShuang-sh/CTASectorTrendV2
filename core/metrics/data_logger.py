"""
Comprehensive Data Logger for Backtesting

Captures all data points for introspection:
- Raw factor values (pre-normalization)
- Signals (post-combination)
- Target and actual positions
- Trades executed
- PnL breakdown
- Volatility targeting effectiveness

Also provides data persistence for:
- Saving/loading signals, trades, positions, metrics
- CSV and Pickle format support
- Run ID based organization
"""

import os
import glob
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json


@dataclass
class DailyLog:
    """单日完整日志记录"""
    date: pd.Timestamp
    factor_values: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, Any] = field(default_factory=dict)
    target_positions: Dict[str, float] = field(default_factory=dict)
    actual_positions: Dict[str, float] = field(default_factory=dict)
    trades: List[Dict] = field(default_factory=list)
    pnl: Dict[str, float] = field(default_factory=dict)
    volatility_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLogger:
    """
    综合数据日志记录器

    功能:
    1. 记录原始因子值 (归一化前)
    2. 记录信号 (组合后)
    3. 记录目标和实际仓位
    4. 记录执行的交易
    5. 记录损益分解
    6. 记录波动率目标执行效果
    7. 支持 IC 指标计算
    """

    def __init__(self, config: Dict = None):
        """
        Args:
            config: 日志配置字典
                - enabled: 是否启用日志
                - output_dir: 输出目录
                - log_levels: 各类型日志的开关
                - run_id: 运行ID (用于持久化文件命名)
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.output_dir = Path(self.config.get('output_dir', '../tests/results/debug'))
        self.run_id = self.config.get('run_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.log_levels = self.config.get('log_levels', {
            'raw_factor_values': True,
            'signals': True,
            'positions': True,
            'trades': True,
            'pnl': True,
            'ic_metrics': True,
            'volatility_targeting': True,
        })

        # 存储
        self.daily_logs: Dict[pd.Timestamp, DailyLog] = {}

        # 缓存用于 IC 计算
        self._factor_history: Dict[str, List[Dict]] = {}
        self._return_history: Dict[str, List[Dict]] = {}

        # 确保输出目录存在
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_factor_values(
        self,
        date: pd.Timestamp,
        factor_outputs: Dict[str, Any]
    ) -> None:
        """
        记录因子计算输出

        Args:
            date: 当前日期
            factor_outputs: {factor_name: FactorOutput} 或 {factor_name: values}
        """
        if not self.enabled or not self.log_levels.get('raw_factor_values', True):
            return

        log = self._get_or_create_log(date)

        for name, output in factor_outputs.items():
            # 支持 FactorOutput 对象或直接的值
            if hasattr(output, 'values'):
                values = output.values
                factor_type = output.factor_type.name if hasattr(output, 'factor_type') else 'UNKNOWN'
                raw_values = output.raw_values if hasattr(output, 'raw_values') else None
            else:
                values = output
                factor_type = 'UNKNOWN'
                raw_values = None

            # 转换为可序列化格式
            if isinstance(values, pd.Series):
                values_dict = values.to_dict()
            elif isinstance(values, dict):
                values_dict = self._convert_dict_keys(values)
            else:
                values_dict = {'value': values}

            log.factor_values[name] = {
                'type': factor_type,
                'values': values_dict,
                'raw_values': raw_values,
            }

            # 更新因子历史用于 IC 计算
            self._update_factor_history(name, date, values_dict)

    def _convert_dict_keys(self, d: Dict) -> Dict:
        """将字典键转换为字符串"""
        result = {}
        for k, v in d.items():
            if isinstance(k, tuple):
                key = f"{k[0]}_{k[1]}"
            else:
                key = str(k)
            result[key] = v
        return result

    def log_signals(
        self,
        date: pd.Timestamp,
        signals: Dict[str, Any]
    ) -> None:
        """
        记录组合后的信号

        Args:
            date: 当前日期
            signals: {ticker_or_pair: CombinedSignal} 或 {ticker: signal_value}
        """
        if not self.enabled or not self.log_levels.get('signals', True):
            return

        log = self._get_or_create_log(date)

        for key, signal in signals.items():
            # 支持 CombinedSignal 对象或直接的值
            if hasattr(signal, 'signal'):
                signal_info = {
                    'signal': signal.signal,
                    'ts_contribution': getattr(signal, 'ts_contribution', 0.0),
                    'xs_global_contribution': getattr(signal, 'xs_global_contribution', 0.0),
                    'xs_pairwise_contribution': getattr(signal, 'xs_pairwise_contribution', 0.0),
                    'pair': signal.pair if hasattr(signal, 'pair') else None,
                }
            else:
                signal_info = {'signal': signal}

            # 转换键为字符串
            str_key = f"{key[0]}_{key[1]}" if isinstance(key, tuple) else str(key)
            log.signals[str_key] = signal_info

    def log_positions(
        self,
        date: pd.Timestamp,
        positions: Dict[str, float],
        position_type: str = 'target'
    ) -> None:
        """
        记录仓位

        Args:
            date: 当前日期
            positions: {ticker: weight/quantity}
            position_type: 'target' 或 'actual'
        """
        if not self.enabled or not self.log_levels.get('positions', True):
            return

        log = self._get_or_create_log(date)

        if position_type == 'target':
            log.target_positions = positions.copy()
        else:
            log.actual_positions = positions.copy()

    def log_trades(
        self,
        date: pd.Timestamp,
        trades: List[Dict]
    ) -> None:
        """
        记录执行的交易

        Args:
            date: 当前日期
            trades: 交易列表 [{ticker, action, quantity, price, ...}, ...]
        """
        if not self.enabled or not self.log_levels.get('trades', True):
            return

        log = self._get_or_create_log(date)
        log.trades = [t.copy() for t in trades]

    def log_pnl(
        self,
        date: pd.Timestamp,
        portfolio: Dict
    ) -> None:
        """
        记录损益分解

        Args:
            date: 当前日期
            portfolio: 投资组合状态字典
        """
        if not self.enabled or not self.log_levels.get('pnl', True):
            return

        log = self._get_or_create_log(date)
        log.pnl = {
            'total_value': portfolio.get('total_value'),
            'cash': portfolio.get('cash'),
            'positions_value': portfolio.get('total_value', 0) - portfolio.get('cash', 0),
            'leverage': portfolio.get('leverage'),
            'daily_return': portfolio.get('daily_return'),
            'cumulative_return': portfolio.get('cumulative_return'),
        }

        # 更新收益历史用于 IC 计算
        self._update_return_history(date, portfolio)

    def log_volatility_targeting(
        self,
        date: pd.Timestamp,
        vol_info: Dict
    ) -> None:
        """
        记录波动率目标执行效果

        Args:
            date: 当前日期
            vol_info: 波动率信息 {target_vol, realized_vol, adjustment_factor, ...}
        """
        if not self.enabled or not self.log_levels.get('volatility_targeting', True):
            return

        log = self._get_or_create_log(date)
        log.volatility_info = vol_info.copy()

    def log_metadata(
        self,
        date: pd.Timestamp,
        metadata: Dict
    ) -> None:
        """记录额外元数据"""
        if not self.enabled:
            return

        log = self._get_or_create_log(date)
        log.metadata.update(metadata)

    def _get_or_create_log(self, date: pd.Timestamp) -> DailyLog:
        """获取或创建日志记录"""
        if date not in self.daily_logs:
            self.daily_logs[date] = DailyLog(date=date)
        return self.daily_logs[date]

    def _update_factor_history(
        self,
        factor_name: str,
        date: pd.Timestamp,
        values: Dict
    ) -> None:
        """更新因子历史记录"""
        if factor_name not in self._factor_history:
            self._factor_history[factor_name] = []

        self._factor_history[factor_name].append({
            'date': date,
            'values': values
        })

    def _update_return_history(
        self,
        date: pd.Timestamp,
        portfolio: Dict
    ) -> None:
        """更新收益历史记录"""
        for ticker, ret in portfolio.get('ticker_returns', {}).items():
            if ticker not in self._return_history:
                self._return_history[ticker] = []
            self._return_history[ticker].append({
                'date': date,
                'return': ret
            })

    def calculate_ic_metrics(
        self,
        forward_periods: List[int] = None
    ) -> Dict[str, Dict]:
        """
        计算因子 IC 指标

        Args:
            forward_periods: 前向收益期数列表，默认 [1, 5, 10, 20]

        Returns:
            {factor_name: {period: {ic, ir, t_stat}}}
        """
        if not self.log_levels.get('ic_metrics', True):
            return {}

        if forward_periods is None:
            forward_periods = [1, 5, 10, 20]

        results = {}

        for factor_name, history in self._factor_history.items():
            if len(history) < 20:
                continue

            factor_results = {}

            for period in forward_periods:
                ic_values = []

                # 对每个日期计算横截面 IC
                for i in range(len(history) - period):
                    factor_date = history[i]['date']
                    factor_values = history[i]['values']

                    # 获取前向收益
                    forward_returns = self._get_forward_returns(
                        factor_date, period, list(factor_values.keys())
                    )

                    if len(forward_returns) < 5:
                        continue

                    # 计算 Spearman IC
                    factor_series = pd.Series(factor_values)
                    return_series = pd.Series(forward_returns)

                    common_idx = factor_series.index.intersection(return_series.index)
                    if len(common_idx) < 5:
                        continue

                    ic = factor_series[common_idx].corr(
                        return_series[common_idx],
                        method='spearman'
                    )
                    if pd.notna(ic):
                        ic_values.append(ic)

                if len(ic_values) >= 10:
                    ic_array = np.array(ic_values)
                    mean_ic = np.mean(ic_array)
                    std_ic = np.std(ic_array)
                    ir = mean_ic / std_ic if std_ic > 0 else 0
                    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_array))) if std_ic > 0 else 0

                    factor_results[f'{period}d'] = {
                        'ic': mean_ic,
                        'ir': ir,
                        't_stat': t_stat,
                        'ic_std': std_ic,
                        'n_obs': len(ic_values)
                    }

            if factor_results:
                results[factor_name] = factor_results

        return results

    def _get_forward_returns(
        self,
        start_date: pd.Timestamp,
        periods: int,
        tickers: List[str]
    ) -> Dict[str, float]:
        """获取前向收益"""
        # 从 return_history 中获取
        returns = {}
        sorted_dates = sorted(self.daily_logs.keys())

        try:
            start_idx = sorted_dates.index(start_date)
            if start_idx + periods >= len(sorted_dates):
                return returns

            end_date = sorted_dates[start_idx + periods]

            for ticker in tickers:
                if ticker in self._return_history:
                    ticker_returns = self._return_history[ticker]
                    # 计算累积收益
                    cumulative = 0.0
                    for rec in ticker_returns:
                        if start_date < rec['date'] <= end_date:
                            cumulative += rec['return']
                    returns[ticker] = cumulative

        except (ValueError, IndexError):
            pass

        return returns

    def get_all_logs(self) -> Dict[str, Dict]:
        """
        导出所有日志

        Returns:
            {date_str: {factor_values, signals, positions, trades, pnl, ...}}
        """
        return {
            str(date): {
                'factor_values': log.factor_values,
                'signals': log.signals,
                'target_positions': log.target_positions,
                'actual_positions': log.actual_positions,
                'trades': log.trades,
                'pnl': log.pnl,
                'volatility_info': log.volatility_info,
                'metadata': log.metadata,
            }
            for date, log in self.daily_logs.items()
        }

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        将日志转换为 DataFrame 格式

        Returns:
            {
                'factor_values': DataFrame,
                'signals': DataFrame,
                'positions': DataFrame,
                'trades': DataFrame,
                'pnl': DataFrame,
            }
        """
        dataframes = {}

        # Factor values
        factor_records = []
        for date, log in sorted(self.daily_logs.items()):
            for factor_name, data in log.factor_values.items():
                for ticker, value in data.get('values', {}).items():
                    factor_records.append({
                        'date': date,
                        'factor': factor_name,
                        'ticker': ticker,
                        'value': value,
                        'type': data.get('type', 'UNKNOWN')
                    })
        if factor_records:
            dataframes['factor_values'] = pd.DataFrame(factor_records)

        # Signals
        signal_records = []
        for date, log in sorted(self.daily_logs.items()):
            for ticker, data in log.signals.items():
                record = {'date': date, 'ticker': ticker}
                record.update(data)
                signal_records.append(record)
        if signal_records:
            dataframes['signals'] = pd.DataFrame(signal_records)

        # Trades
        trade_records = []
        for date, log in sorted(self.daily_logs.items()):
            for trade in log.trades:
                record = {'date': date}
                record.update(trade)
                trade_records.append(record)
        if trade_records:
            dataframes['trades'] = pd.DataFrame(trade_records)

        # PnL
        pnl_records = []
        for date, log in sorted(self.daily_logs.items()):
            if log.pnl:
                record = {'date': date}
                record.update(log.pnl)
                pnl_records.append(record)
        if pnl_records:
            dataframes['pnl'] = pd.DataFrame(pnl_records)

        return dataframes

    def export_to_excel(self, filepath: str = None) -> str:
        """
        导出日志到 Excel 文件

        Args:
            filepath: 输出文件路径，默认自动生成

        Returns:
            实际保存的文件路径
        """
        if filepath is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f'backtest_logs_{timestamp}.xlsx'

        dataframes = self.to_dataframes()

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for name, df in dataframes.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=name[:31], index=False)

        return str(filepath)

    def export_to_json(self, filepath: str = None) -> str:
        """
        导出日志到 JSON 文件

        Args:
            filepath: 输出文件路径

        Returns:
            实际保存的文件路径
        """
        if filepath is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f'backtest_logs_{timestamp}.json'

        logs = self.get_all_logs()

        # 转换日期键为字符串
        serializable_logs = {}
        for date, data in logs.items():
            serializable_logs[str(date)] = data

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_logs, f, ensure_ascii=False, indent=2, default=str)

        return str(filepath)

    def clear(self) -> None:
        """清空所有日志"""
        self.daily_logs.clear()
        self._factor_history.clear()
        self._return_history.clear()

    # ========== 持久化方法 (从 Persistence 模块整合) ==========

    def save_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        factor_name: str,
        scheme: str,
        pair: Tuple[str, str] = None
    ) -> List[str]:
        """
        保存信号数据到文件

        Args:
            signals: {industry: signal_df} 信号字典
            factor_name: 因子名称
            scheme: 方案名称
            pair: 可选的配对标识

        Returns:
            保存的文件路径列表
        """
        if not self.enabled:
            return []

        saved_files = []
        pair_str = f"_{pair[0]}_{pair[1]}" if pair else ""
        prefix = f"{factor_name}_{scheme}{pair_str}_{self.run_id}"

        # CSV 格式 - 按行业分别保存
        for industry, df in signals.items():
            if not df.empty:
                filename = self.output_dir / f"{prefix}_{industry}_signals.csv"
                df.to_csv(filename, index=False)
                saved_files.append(str(filename))

        # Pickle 格式 - 完整保存
        pkl_filename = self.output_dir / f"{prefix}_signals.pkl"
        with open(pkl_filename, 'wb') as f:
            pickle.dump(signals, f)
        saved_files.append(str(pkl_filename))

        return saved_files

    def save_trades_to_file(
        self,
        trades: List[Dict],
        prefix: str = ""
    ) -> Optional[str]:
        """
        保存交易明细到 CSV 文件

        Args:
            trades: 交易记录列表
            prefix: 文件名前缀

        Returns:
            保存的文件路径，如果没有数据则返回 None
        """
        if not self.enabled or not trades:
            return None

        df = pd.DataFrame(trades)
        filename = self.output_dir / f"{prefix}_{self.run_id}_trades.csv"
        df.to_csv(filename, index=False)
        return str(filename)

    def save_positions_to_file(
        self,
        positions: List[Dict],
        prefix: str = ""
    ) -> Optional[str]:
        """
        保存持仓快照到 CSV 文件

        Args:
            positions: 持仓记录列表
            prefix: 文件名前缀

        Returns:
            保存的文件路径
        """
        if not self.enabled or not positions:
            return None

        df = pd.DataFrame(positions)
        filename = self.output_dir / f"{prefix}_{self.run_id}_positions.csv"
        df.to_csv(filename, index=False)
        return str(filename)

    def save_metrics(
        self,
        metrics: Dict[str, Any],
        prefix: str = ""
    ) -> Optional[str]:
        """
        保存绩效指标到 Pickle 文件

        Args:
            metrics: 绩效指标字典
            prefix: 文件名前缀

        Returns:
            保存的文件路径
        """
        if not self.enabled:
            return None

        filename = self.output_dir / f"{prefix}_{self.run_id}_metrics.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(metrics, f)
        return str(filename)

    def load_signals(
        self,
        factor_name: str,
        scheme: str
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        加载之前保存的信号数据

        Args:
            factor_name: 因子名称
            scheme: 方案名称

        Returns:
            信号字典，如果未找到则返回 None
        """
        pattern = f"{factor_name}_{scheme}_*_signals.pkl"
        files = glob.glob(str(self.output_dir / pattern))

        if files:
            # 返回最新的文件
            with open(sorted(files)[-1], 'rb') as f:
                return pickle.load(f)
        return None

    def load_metrics(
        self,
        factor_name: str = None,
        scheme: str = None,
        prefix: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        加载之前保存的绩效指标

        Args:
            factor_name: 因子名称 (可选)
            scheme: 方案名称 (可选)
            prefix: 文件名前缀 (可选)

        Returns:
            绩效指标字典，如果未找到则返回 None
        """
        if factor_name and scheme:
            pattern = f"{factor_name}_{scheme}_*_metrics.pkl"
        elif prefix:
            pattern = f"{prefix}_*_metrics.pkl"
        else:
            pattern = "*_metrics.pkl"

        files = glob.glob(str(self.output_dir / pattern))

        if files:
            with open(sorted(files)[-1], 'rb') as f:
                return pickle.load(f)
        return None

    def save_all(
        self,
        signals: Dict[str, pd.DataFrame] = None,
        trades: List[Dict] = None,
        positions: List[Dict] = None,
        metrics: Dict[str, Any] = None,
        factor_name: str = "unknown",
        scheme: str = "A"
    ) -> Dict[str, str]:
        """
        一次性保存所有数据

        Args:
            signals: 信号字典
            trades: 交易记录
            positions: 持仓记录
            metrics: 绩效指标
            factor_name: 因子名称
            scheme: 方案名称

        Returns:
            {数据类型: 文件路径} 字典
        """
        saved = {}
        prefix = f"{factor_name}_{scheme}"

        if signals:
            files = self.save_signals(signals, factor_name, scheme)
            if files:
                saved['signals'] = files[-1]  # pkl 文件

        if trades:
            path = self.save_trades_to_file(trades, prefix)
            if path:
                saved['trades'] = path

        if positions:
            path = self.save_positions_to_file(positions, prefix)
            if path:
                saved['positions'] = path

        if metrics:
            path = self.save_metrics(metrics, prefix)
            if path:
                saved['metrics'] = path

        # 同时导出日志
        if self.daily_logs:
            excel_path = self.export_to_excel()
            saved['logs_excel'] = excel_path

        return saved

    def __len__(self) -> int:
        return len(self.daily_logs)

    def __repr__(self) -> str:
        return f"DataLogger(days={len(self)}, enabled={self.enabled})"
