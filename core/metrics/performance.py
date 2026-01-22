"""
Performance Metrics Module

Provides comprehensive strategy performance calculation:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Trade statistics
- Rolling metrics
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceResult:
    """
    绩效结果数据类

    Attributes:
        metrics: 核心绩效指标字典
        equity_curve: 权益曲线
        drawdown_series: 回撤序列
        monthly_returns: 月度收益表
    """
    metrics: Dict[str, float]
    equity_curve: pd.Series = None
    drawdown_series: pd.Series = None
    monthly_returns: pd.DataFrame = None


class PerformanceMetrics:
    """
    绩效指标计算器

    功能:
    - 风险调整收益: Sharpe, Sortino, Calmar
    - 回撤分析: 最大回撤、回撤持续时间
    - 交易统计: 胜率、盈亏比、平均持仓时间
    - 滚动指标: 滚动 Sharpe、滚动波动率
    """

    def __init__(self, risk_free_rate: float = 0.0, annualization_factor: int = 252):
        """
        Args:
            risk_free_rate: 无风险利率 (年化)
            annualization_factor: 年化因子 (交易日数)
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def calculate_all(
        self,
        equity_curve: pd.Series,
        trades: List[Dict] = None,
        returns: pd.Series = None
    ) -> PerformanceResult:
        """
        计算所有绩效指标

        Args:
            equity_curve: 权益曲线 (以日期为索引)
            trades: 交易列表 [{pnl, holding_days, ...}, ...]
            returns: 收益率序列 (可选，若不提供则从权益曲线计算)

        Returns:
            PerformanceResult 包含所有指标
        """
        if returns is None:
            returns = equity_curve.pct_change().dropna()

        # 核心指标
        metrics = {}

        # 收益指标
        metrics['total_return'] = self.calculate_total_return(equity_curve)
        metrics['annual_return'] = self.calculate_annual_return(returns)
        metrics['annual_volatility'] = self.calculate_annual_volatility(returns)

        # 风险调整收益
        metrics['sharpe_ratio'] = self.calculate_sharpe(returns)
        metrics['sortino_ratio'] = self.calculate_sortino(returns)

        # 回撤指标
        dd_result = self.calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = dd_result[0]
        metrics['max_drawdown_start'] = dd_result[1]
        metrics['max_drawdown_end'] = dd_result[2]
        metrics['calmar_ratio'] = self.calculate_calmar(returns, dd_result[0])

        # 回撤序列
        drawdown_series = self.calculate_drawdown_series(equity_curve)

        # 交易统计
        if trades:
            trade_stats = self.calculate_trade_statistics(trades)
            metrics.update(trade_stats)

        # 月度收益
        monthly_returns = self.calculate_monthly_returns(returns)

        # 其他风险指标
        metrics['skewness'] = returns.skew() if len(returns) > 2 else 0.0
        metrics['kurtosis'] = returns.kurtosis() if len(returns) > 2 else 0.0
        metrics['var_95'] = self.calculate_var(returns, confidence=0.95)
        metrics['cvar_95'] = self.calculate_cvar(returns, confidence=0.95)

        return PerformanceResult(
            metrics=metrics,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            monthly_returns=monthly_returns
        )

    @staticmethod
    def calculate_total_return(equity_curve: pd.Series) -> float:
        """计算总收益率"""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    def calculate_annual_return(self, returns: pd.Series) -> float:
        """计算年化收益率"""
        if len(returns) < 2:
            return 0.0
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / self.annualization_factor
        if n_years <= 0:
            return 0.0
        return (1 + total_return) ** (1 / n_years) - 1

    def calculate_annual_volatility(self, returns: pd.Series) -> float:
        """计算年化波动率"""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(self.annualization_factor)

    def calculate_sharpe(self, returns: pd.Series) -> float:
        """
        计算 Sharpe 比率

        Sharpe = (Annual Return - Risk Free Rate) / Annual Volatility
        """
        if len(returns) < 2:
            return 0.0

        annual_return = self.calculate_annual_return(returns)
        annual_vol = self.calculate_annual_volatility(returns)

        if annual_vol == 0:
            return 0.0

        return (annual_return - self.risk_free_rate) / annual_vol

    def calculate_sortino(self, returns: pd.Series) -> float:
        """
        计算 Sortino 比率

        Sortino = (Annual Return - Risk Free Rate) / Downside Volatility
        """
        if len(returns) < 2:
            return 0.0

        annual_return = self.calculate_annual_return(returns)

        # 下行波动率
        negative_returns = returns[returns < 0]
        if len(negative_returns) < 1:
            return float('inf') if annual_return > self.risk_free_rate else 0.0

        downside_vol = negative_returns.std() * np.sqrt(self.annualization_factor)

        if downside_vol == 0:
            return 0.0

        return (annual_return - self.risk_free_rate) / downside_vol

    @staticmethod
    def calculate_max_drawdown(
        equity_curve: pd.Series
    ) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        计算最大回撤

        Returns:
            (max_drawdown, peak_date, trough_date)
        """
        if len(equity_curve) < 2:
            return 0.0, None, None

        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max

        max_dd = drawdown.min()
        if pd.isna(max_dd) or max_dd >= 0:
            return 0.0, None, None

        # 找到最大回撤的谷点
        trough_idx = drawdown.idxmin()

        # 找到对应的峰点
        peak_idx = equity_curve[:trough_idx].idxmax()

        return abs(max_dd), peak_idx, trough_idx

    @staticmethod
    def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """计算回撤序列"""
        if len(equity_curve) < 2:
            return pd.Series(dtype=float)

        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown

    def calculate_calmar(
        self,
        returns: pd.Series,
        max_drawdown: float
    ) -> float:
        """
        计算 Calmar 比率

        Calmar = Annual Return / Max Drawdown
        """
        if max_drawdown == 0:
            return 0.0

        annual_return = self.calculate_annual_return(returns)
        return annual_return / max_drawdown

    @staticmethod
    def calculate_trade_statistics(trades: List[Dict]) -> Dict[str, float]:
        """
        计算交易统计

        Args:
            trades: 交易列表，每个交易包含 'pnl', 'holding_days' 等字段

        Returns:
            交易统计指标字典
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'avg_holding_days': 0.0,
            }

        # 分离盈利和亏损交易
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        total_trades = len(pnls)
        winning_trades = len(winning)
        losing_trades = len(losing)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(winning) if winning else 0.0
        avg_loss = abs(np.mean(losing)) if losing else 0.0

        # 盈亏比
        profit_factor = avg_win / avg_loss if avg_loss > 0 else (float('inf') if avg_win > 0 else 0.0)

        # 平均持仓天数
        holding_days = [t.get('holding_days', 0) for t in trades if 'holding_days' in t]
        avg_holding = np.mean(holding_days) if holding_days else 0.0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding,
        }

    @staticmethod
    def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
        """
        计算月度收益表

        Returns:
            月度收益 DataFrame (行=年, 列=月)
        """
        if len(returns) < 2:
            return pd.DataFrame()

        if not isinstance(returns.index, pd.DatetimeIndex):
            return pd.DataFrame()

        # 按月聚合
        monthly = (1 + returns).resample('M').prod() - 1

        # 创建年-月矩阵
        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month

        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

        return pivot

    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        计算 Value at Risk (历史模拟法)

        Args:
            returns: 收益率序列
            confidence: 置信水平

        Returns:
            VaR 值 (正数表示潜在损失)
        """
        if len(returns) < 10:
            return 0.0
        return -np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        计算 Conditional VaR (Expected Shortfall)

        Args:
            returns: 收益率序列
            confidence: 置信水平

        Returns:
            CVaR 值 (正数表示潜在损失)
        """
        if len(returns) < 10:
            return 0.0
        var = np.percentile(returns, (1 - confidence) * 100)
        return -returns[returns <= var].mean()

    def calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        计算滚动 Sharpe 比率

        Args:
            returns: 收益率序列
            window: 滚动窗口 (交易日)

        Returns:
            滚动 Sharpe 序列
        """
        if len(returns) < window:
            return pd.Series(dtype=float)

        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        # 年化
        annualized_return = rolling_mean * self.annualization_factor
        annualized_vol = rolling_std * np.sqrt(self.annualization_factor)

        return (annualized_return - self.risk_free_rate) / annualized_vol

    def calculate_rolling_volatility(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        计算滚动波动率

        Args:
            returns: 收益率序列
            window: 滚动窗口

        Returns:
            滚动年化波动率序列
        """
        if len(returns) < window:
            return pd.Series(dtype=float)

        return returns.rolling(window).std() * np.sqrt(self.annualization_factor)

    def generate_report(
        self,
        result: PerformanceResult,
        name: str = "Strategy"
    ) -> str:
        """
        生成绩效报告文本

        Args:
            result: PerformanceResult 对象
            name: 策略名称

        Returns:
            格式化的报告文本
        """
        m = result.metrics
        lines = [
            f"{'=' * 60}",
            f"{name} 绩效报告",
            f"{'=' * 60}",
            "",
            "【收益指标】",
            f"  总收益率:     {m.get('total_return', 0) * 100:.2f}%",
            f"  年化收益率:   {m.get('annual_return', 0) * 100:.2f}%",
            f"  年化波动率:   {m.get('annual_volatility', 0) * 100:.2f}%",
            "",
            "【风险调整收益】",
            f"  Sharpe 比率:  {m.get('sharpe_ratio', 0):.3f}",
            f"  Sortino 比率: {m.get('sortino_ratio', 0):.3f}",
            f"  Calmar 比率:  {m.get('calmar_ratio', 0):.3f}",
            "",
            "【回撤分析】",
            f"  最大回撤:     {m.get('max_drawdown', 0) * 100:.2f}%",
            "",
            "【风险指标】",
            f"  VaR (95%):    {m.get('var_95', 0) * 100:.2f}%",
            f"  CVaR (95%):   {m.get('cvar_95', 0) * 100:.2f}%",
            f"  偏度:         {m.get('skewness', 0):.3f}",
            f"  峰度:         {m.get('kurtosis', 0):.3f}",
        ]

        # 交易统计 (如果有)
        if 'total_trades' in m:
            lines.extend([
                "",
                "【交易统计】",
                f"  总交易次数:   {m.get('total_trades', 0)}",
                f"  盈利交易:     {m.get('winning_trades', 0)}",
                f"  亏损交易:     {m.get('losing_trades', 0)}",
                f"  胜率:         {m.get('win_rate', 0) * 100:.1f}%",
                f"  平均盈利:     {m.get('avg_win', 0):.2f}",
                f"  平均亏损:     {m.get('avg_loss', 0):.2f}",
                f"  盈亏比:       {m.get('profit_factor', 0):.2f}",
                f"  平均持仓天数: {m.get('avg_holding_days', 0):.1f}",
            ])

        lines.append(f"{'=' * 60}")

        return "\n".join(lines)
