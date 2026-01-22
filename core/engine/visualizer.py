#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Backtest Visualization

Visualization utilities for backtest results:
- Equity curves
- Drawdown analysis
- Monthly returns heatmap
- Trade analysis charts
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np
from datetime import datetime

# Conditional matplotlib imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

if TYPE_CHECKING:
    from core.engine.backtest_engine import BacktestResult


def _setup_chinese_font():
    """Setup Chinese font for matplotlib."""
    if not HAS_MATPLOTLIB:
        return

    import platform
    system = platform.system()

    if system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'STHeiti']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']

    plt.rcParams['axes.unicode_minus'] = False


class BacktestVisualizer:
    """
    Backtest result visualizer.

    Provides multiple visualization methods:
    - plot_results: Comprehensive results (equity curve, drawdown, leverage)
    - plot_monthly_returns_heatmap: Monthly returns heatmap
    - plot_trade_analysis: Trade analysis charts
    - plot_drawdown: Detailed drawdown analysis
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        _setup_chinese_font()

    def plot_results(
        self,
        result: 'BacktestResult',
        title: str = 'Backtest Results',
        save_path: str = None,
        show: bool = True
    ) -> Optional['plt.Figure']:
        """
        Plot comprehensive results.

        Includes:
        - Equity curve
        - Drawdown curve
        - Leverage changes

        Args:
            result: BacktestResult object
            title: Chart title
            save_path: Optional save path
            show: Whether to display

        Returns:
            matplotlib Figure or None
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed, cannot plot")
            return None

        equity_curve = result.equity_curve
        if equity_curve is None or equity_curve.empty:
            print("Warning: Empty equity curve")
            return None

        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=True)

        # 1. Equity curve
        ax1 = axes[0]
        ax1.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=1.5, label='Portfolio Value')
        ax1.axhline(y=equity_curve.iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title(title)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))

        # 2. Drawdown curve
        ax2 = axes[1]
        drawdown = self._calculate_drawdown(equity_curve)
        ax2.fill_between(drawdown.index, 0, drawdown.values * 100, color='red', alpha=0.3, label='Drawdown')
        ax2.plot(drawdown.index, drawdown.values * 100, 'r-', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)

        # 3. Leverage
        ax3 = axes[2]
        if result.portfolio_history:
            leverage_data = pd.Series({
                h['date']: h.get('leverage', 0)
                for h in result.portfolio_history
                if h.get('date') is not None
            })
            if not leverage_data.empty:
                ax3.plot(leverage_data.index, leverage_data.values, 'g-', linewidth=1, label='Leverage')
                ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
                ax3.set_ylabel('Leverage')
                ax3.legend(loc='upper left')
                ax3.grid(True, alpha=0.3)

        ax3.set_xlabel('Date')

        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_monthly_returns_heatmap(
        self,
        result: 'BacktestResult',
        title: str = 'Monthly Returns Heatmap',
        save_path: str = None,
        show: bool = True
    ) -> Optional['plt.Figure']:
        """
        Plot monthly returns heatmap.

        Args:
            result: BacktestResult object
            title: Chart title
            save_path: Optional save path
            show: Whether to display

        Returns:
            matplotlib Figure or None
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            print("Warning: matplotlib/seaborn not installed, cannot plot heatmap")
            return None

        equity_curve = result.equity_curve
        if equity_curve is None or equity_curve.empty:
            print("Warning: Empty equity curve")
            return None

        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()

        if monthly_returns.empty:
            print("Warning: Empty monthly returns data")
            return None

        returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values * 100
        })

        pivot = returns_df.pivot(index='year', columns='month', values='return')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = month_names[:len(pivot.columns)]

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Return (%)'}
        )

        ax.set_title(title)
        ax.set_ylabel('Year')
        ax.set_xlabel('Month')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Heatmap saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_trade_analysis(
        self,
        trades: List[Dict],
        title: str = 'Trade Analysis',
        save_path: str = None,
        show: bool = True
    ) -> Optional['plt.Figure']:
        """
        Plot trade analysis charts.

        Includes:
        - P&L distribution histogram
        - Holding period distribution
        - Cumulative P&L
        - Win rate pie chart

        Args:
            trades: List of trade records
            title: Chart title
            save_path: Optional save path
            show: Whether to display

        Returns:
            matplotlib Figure or None
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed, cannot plot")
            return None

        if not trades:
            print("Warning: Empty trade records")
            return None

        trades_df = pd.DataFrame(trades)

        pnl_trades = trades_df[
            (trades_df.get('pnl', pd.Series([0] * len(trades_df))) != 0) |
            (trades_df.get('action', pd.Series([''] * len(trades_df))).isin(['CLOSE', 'FLIP']))
        ].copy()

        if pnl_trades.empty:
            if 'pnl' in trades_df.columns:
                pnl_trades = trades_df[trades_df['pnl'].notna()].copy()

        if pnl_trades.empty:
            print("Warning: No P&L trade records")
            return None

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # 1. P&L distribution
        ax1 = axes[0, 0]
        if 'pnl' in pnl_trades.columns:
            pnl_values = pnl_trades['pnl'].dropna()
            if not pnl_values.empty:
                ax1.hist(pnl_values, bins=30, color='blue', alpha=0.7, edgecolor='black')
                ax1.axvline(x=0, color='black', linestyle='--')
                ax1.axvline(x=pnl_values.mean(), color='orange', linestyle='-',
                            label=f'Mean: {pnl_values.mean():.0f}')
                ax1.set_xlabel('P&L')
                ax1.set_ylabel('Frequency')
                ax1.set_title('P&L Distribution')
                ax1.legend()

        # 2. Holding period distribution
        ax2 = axes[0, 1]
        if 'holding_days' in pnl_trades.columns:
            holding_days = pnl_trades['holding_days'].dropna()
            if not holding_days.empty:
                ax2.hist(holding_days, bins=20, color='purple', alpha=0.7, edgecolor='black')
                ax2.axvline(x=holding_days.mean(), color='orange', linestyle='-',
                            label=f'Mean: {holding_days.mean():.1f} days')
                ax2.set_xlabel('Holding Days')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Holding Period Distribution')
                ax2.legend()

        # 3. Cumulative P&L
        ax3 = axes[1, 0]
        if 'pnl' in pnl_trades.columns and 'date' in pnl_trades.columns:
            pnl_trades_sorted = pnl_trades.sort_values('date')
            cumulative_pnl = pnl_trades_sorted['pnl'].cumsum()
            ax3.plot(range(len(cumulative_pnl)), cumulative_pnl.values, 'b-', linewidth=1.5)
            ax3.axhline(y=0, color='gray', linestyle='--')
            ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl.values,
                             where=cumulative_pnl.values > 0, color='green', alpha=0.3)
            ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl.values,
                             where=cumulative_pnl.values < 0, color='red', alpha=0.3)
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('Cumulative P&L')
            ax3.set_title('Cumulative P&L Curve')

        # 4. Win rate pie chart
        ax4 = axes[1, 1]
        if 'pnl' in pnl_trades.columns:
            wins = (pnl_trades['pnl'] > 0).sum()
            losses = (pnl_trades['pnl'] < 0).sum()
            ties = (pnl_trades['pnl'] == 0).sum()

            if wins + losses > 0:
                sizes = [wins, losses]
                labels = [f'Wins ({wins})', f'Losses ({losses})']
                colors_pie = ['green', 'red']

                if ties > 0:
                    sizes.append(ties)
                    labels.append(f'Ties ({ties})')
                    colors_pie.append('gray')

                ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
                ax4.set_title(f'Win Rate: {wins / (wins + losses) * 100:.1f}%')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trade analysis saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_drawdown(
        self,
        equity_curve: pd.Series,
        title: str = 'Drawdown Analysis',
        save_path: str = None,
        show: bool = True
    ) -> Optional['plt.Figure']:
        """
        Plot detailed drawdown analysis.

        Args:
            equity_curve: Equity curve Series
            title: Chart title
            save_path: Optional save path
            show: Whether to display

        Returns:
            matplotlib Figure or None
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not installed, cannot plot")
            return None

        if equity_curve is None or equity_curve.empty:
            print("Warning: Empty equity curve")
            return None

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        drawdown = self._calculate_drawdown(equity_curve)
        peak = equity_curve.expanding().max()

        # 1. Equity curve with peak
        ax1 = axes[0]
        ax1.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=1.5, label='Portfolio Value')
        ax1.plot(peak.index, peak.values, 'g--', linewidth=1, alpha=0.7, label='Peak')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title(title)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))

        # 2. Drawdown curve
        ax2 = axes[1]
        ax2.fill_between(drawdown.index, 0, drawdown.values * 100, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values * 100, 'r-', linewidth=1)

        # Annotate max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax2.annotate(
            f'Max DD: {max_dd_val * 100:.1f}%',
            xy=(max_dd_idx, max_dd_val * 100),
            xytext=(max_dd_idx, max_dd_val * 100 - 5),
            fontsize=10,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='black')
        )

        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Drawdown analysis saved: {save_path}")

        if show:
            plt.show()

        return fig

    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown

    def save_all_plots(
        self,
        result: 'BacktestResult',
        output_dir: str,
        prefix: str = 'backtest'
    ) -> Dict[str, str]:
        """
        Save all plots to files.

        Args:
            result: BacktestResult object
            output_dir: Output directory
            prefix: File name prefix

        Returns:
            Dict of {plot_type: file_path}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        files = {}

        # Results chart
        results_path = output_path / f'{prefix}_results_{timestamp}.png'
        self.plot_results(result, save_path=str(results_path), show=False)
        files['results'] = str(results_path)

        # Monthly heatmap
        heatmap_path = output_path / f'{prefix}_monthly_heatmap_{timestamp}.png'
        self.plot_monthly_returns_heatmap(result, save_path=str(heatmap_path), show=False)
        files['monthly_heatmap'] = str(heatmap_path)

        # Drawdown analysis
        drawdown_path = output_path / f'{prefix}_drawdown_{timestamp}.png'
        if result.equity_curve is not None:
            self.plot_drawdown(result.equity_curve, save_path=str(drawdown_path), show=False)
            files['drawdown'] = str(drawdown_path)

        # Trade analysis
        if result.trade_history:
            trades_path = output_path / f'{prefix}_trades_{timestamp}.png'
            self.plot_trade_analysis(result.trade_history, save_path=str(trades_path), show=False)
            files['trades'] = str(trades_path)

        return files


def create_visualizer(figsize: Tuple[int, int] = (14, 10)) -> BacktestVisualizer:
    """Factory function to create a visualizer."""
    return BacktestVisualizer(figsize=figsize)
