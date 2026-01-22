#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTASectorTrendV2 - Unified Entry Point

Supports 7 backtest modes:
1. single_factor_single_asset_ts  - 单因子-单资产-时序
2. multi_factor_single_asset_ts   - 多因子-单资产-时序
3. single_factor_multi_asset_xs   - 单因子-多资产-截面
4. multi_factor_multi_asset_xs    - 多因子-多资产-截面
5. single_factor_single_pair      - 单因子-单对资产-配对
6. multi_factor_single_pair       - 多因子-单对资产-配对
7. multi_factor_multi_asset_comprehensive - 多因子-多资产-综合

Usage:
    # Run from configuration file
    python run.py --config config/my_task.yaml

    # Quick run with mode and assets
    python run.py --mode single_factor_single_asset_ts --asset RB --factor HurstExponent

    # List available modes and factors
    python run.py --list-modes
    python run.py --list-factors
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='CTASectorTrendV2 - Quantitative Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from config file
  python run.py --config config/my_task.yaml

  # Quick single factor single asset
  python run.py --mode single_factor_single_asset_ts --asset RB --factor HurstExponent

  # Multi-factor comprehensive
  python run.py --mode multi_factor_multi_asset_comprehensive --sectors Wind煤焦钢矿

  # Pair trading
  python run.py --mode single_factor_single_pair --pair RB HC --factor CopulaPairFactor

  # List available options
  python run.py --list-modes
  python run.py --list-factors
        """
    )

    # Configuration file
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to task configuration YAML file'
    )

    # Quick run options
    parser.add_argument(
        '--mode', '-m',
        type=str,
        help='Backtest mode (use --list-modes to see options)'
    )

    parser.add_argument(
        '--asset', '-a',
        type=str,
        help='Single asset code (e.g., RB)'
    )

    parser.add_argument(
        '--assets',
        type=str,
        nargs='+',
        help='Multiple asset codes (e.g., RB HC I)'
    )

    parser.add_argument(
        '--sectors',
        type=str,
        nargs='+',
        help='Sector filters (e.g., Wind煤焦钢矿 Wind有色)'
    )

    parser.add_argument(
        '--pair',
        type=str,
        nargs=2,
        help='Pair for pair trading mode (e.g., RB HC)'
    )

    parser.add_argument(
        '--pairs',
        type=str,
        nargs='+',
        help='Multiple pairs (e.g., RB,HC I,J)'
    )

    parser.add_argument(
        '--factor', '-f',
        type=str,
        help='Single factor name'
    )

    parser.add_argument(
        '--factors',
        type=str,
        nargs='+',
        help='Multiple factor names'
    )

    # Date range
    parser.add_argument(
        '--start',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD)'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./Result',
        help='Output directory'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to files'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots'
    )

    # Utility options
    parser.add_argument(
        '--list-modes',
        action='store_true',
        help='List all available backtest modes'
    )

    parser.add_argument(
        '--list-factors',
        action='store_true',
        help='List all available factors'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.list_modes:
        list_modes()
        return

    if args.list_factors:
        list_factors()
        return

    # Determine run mode
    if args.config:
        run_from_config(args)
    elif args.mode:
        run_quick_mode(args)
    else:
        print("Error: Must specify --config or --mode")
        print("Use --list-modes to see available modes")
        print("Use --help for usage information")
        sys.exit(1)


def list_modes():
    """List all available backtest modes."""
    print("\n" + "=" * 60)
    print("Available Backtest Modes")
    print("=" * 60)

    modes = [
        ("single_factor_single_asset_ts", "单因子-单资产-时序", "Single factor applied to single asset with time-series signals"),
        ("multi_factor_single_asset_ts", "多因子-单资产-时序", "Multiple factors combined for single asset"),
        ("single_factor_multi_asset_xs", "单因子-多资产-截面", "Single factor with cross-sectional ranking"),
        ("multi_factor_multi_asset_xs", "多因子-多资产-截面", "Multiple factors with cross-sectional ranking"),
        ("single_factor_single_pair", "单因子-单对资产-配对", "Single pair trading factor"),
        ("multi_factor_single_pair", "多因子-单对资产-配对", "Multiple pair trading factors combined"),
        ("multi_factor_multi_asset_comprehensive", "多因子-多资产-综合", "Full comprehensive multi-factor strategy"),
    ]

    for mode, cn_name, desc in modes:
        print(f"\n{mode}")
        print(f"  中文: {cn_name}")
        print(f"  描述: {desc}")

    print("\n" + "=" * 60)
    print("\nUsage examples:")
    print("  python run.py --mode single_factor_single_asset_ts --asset RB --factor HurstExponent")
    print("  python run.py --mode multi_factor_multi_asset_comprehensive --sectors Wind煤焦钢矿")
    print()


def list_factors():
    """List all available factors."""
    print("\n" + "=" * 60)
    print("Available Factors in CTASectorTrendV2")
    print("=" * 60)

    from core.factors.registry import get_registry
    from core.factors.base import FactorType

    registry = get_registry()

    # Register known factors
    _register_all_factors()

    print("\n[Time Series Factors] 时序因子")
    ts_factors = registry.list_factors(FactorType.TIME_SERIES)
    for name in ts_factors:
        print(f"  - {name}")

    if not ts_factors:
        print("  (Import factors to register them)")

    print("\n[Cross-Sectional Factors] 截面因子")
    xs_factors = registry.list_factors(FactorType.XS_GLOBAL)
    for name in xs_factors:
        print(f"  - {name}")

    if not xs_factors:
        print("  (No XS_GLOBAL factors registered)")

    print("\n[Pair Trading Factors] 配对因子")
    pair_factors = registry.list_factors(FactorType.XS_PAIRWISE)
    for name in pair_factors:
        print(f"  - {name}")

    if not pair_factors:
        print("  (Import factors to register them)")

    print("\n" + "=" * 60)


def _register_all_factors():
    """Register all factors to the registry."""
    from core.factors.registry import register_factor

    # Time Series Factors
    register_factor(
        'HurstExponent',
        class_path='core.factors.time_series.trend.HurstExponent'
    )
    register_factor(
        'EMDTrend',
        class_path='core.factors.time_series.trend.EMDTrend'
    )
    register_factor(
        'AMIHUDFactor',
        class_path='core.factors.time_series.liquidity.AMIHUDFactor'
    )
    register_factor(
        'AmivestFactor',
        class_path='core.factors.time_series.liquidity.AmivestFactor'
    )

    # Pair Trading Factors
    register_factor(
        'CopulaPairFactor',
        class_path='core.factors.pair_trading.copula.CopulaPairFactor'
    )


def run_from_config(args):
    """Run backtest from configuration file."""
    from core.engine import BacktestRunner

    config_path = args.config
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    verbose = not args.quiet

    print(f"\n{'='*60}")
    print("CTASectorTrendV2 - Running from Configuration")
    print(f"{'='*60}")
    print(f"Config: {config_path}")

    runner = BacktestRunner()
    result = runner.run_from_config(config_path, verbose=verbose)

    # Print summary
    if verbose:
        _print_result_summary(result)

    # Save results if requested
    if args.save:
        _save_results(result, args.output, Path(config_path).stem)


def run_quick_mode(args):
    """Run backtest with quick command-line options."""
    from core.engine import BacktestRunner, TaskMode, create_task_config
    import pandas as pd

    verbose = not args.quiet

    # Parse mode
    try:
        mode = TaskMode.from_string(args.mode)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-modes to see available modes")
        sys.exit(1)

    # Build assets list
    assets = []
    if args.asset:
        assets = [args.asset]
    elif args.assets:
        assets = args.assets

    # Build pairs list
    pairs = []
    if args.pair:
        pairs = [tuple(args.pair)]
    elif args.pairs:
        pairs = [tuple(p.split(',')) for p in args.pairs]

    # Build factors config
    factors = {'time_series': [], 'cross_sectional': [], 'pair_trading': []}

    factor_names = []
    if args.factor:
        factor_names = [args.factor]
    elif args.factors:
        factor_names = args.factors

    # Categorize factors (simple heuristic)
    for name in factor_names:
        if 'Pair' in name or 'Copula' in name or 'Kalman' in name:
            factors['pair_trading'].append({'name': name, 'enabled': True})
        elif 'Momentum' in name or 'Rank' in name:
            factors['cross_sectional'].append({'name': name, 'enabled': True})
        else:
            factors['time_series'].append({'name': name, 'enabled': True})

    # Validate mode requirements
    if mode.is_single_asset() and not assets:
        print("Error: Single asset mode requires --asset")
        sys.exit(1)

    if mode.is_pair_trading() and not pairs:
        print("Error: Pair trading mode requires --pair")
        sys.exit(1)

    # Create task config
    task_config = create_task_config(
        mode=args.mode,
        assets=assets,
        pairs=pairs,
        factors=factors,
        name=f"quick_{args.mode}",
        start_date=args.start,
        end_date=args.end,
        output_config={
            'result_dir': args.output,
            'generate_plots': args.plot
        }
    )

    # Add sectors if specified
    if args.sectors:
        task_config.sectors = args.sectors

    print(f"\n{'='*60}")
    print("CTASectorTrendV2 - Quick Run")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Assets: {assets or 'Auto'}")
    print(f"Pairs: {pairs or 'N/A'}")
    print(f"Factors: {factor_names or 'Default'}")
    print(f"Period: {args.start} to {args.end}")
    print(f"{'='*60}\n")

    # Run backtest
    runner = BacktestRunner()
    result = runner.run(task_config, verbose=verbose)

    # Print summary
    if verbose:
        _print_result_summary(result)

    # Save results if requested
    if args.save:
        _save_results(result, args.output, f"quick_{args.mode}")


def _print_result_summary(result):
    """Print result summary."""
    print(f"\n{'='*60}")
    print("Backtest Results Summary")
    print(f"{'='*60}")

    metrics = result.performance_metrics

    if metrics:
        print(f"\nPerformance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'ratio' in key.lower() or 'sharpe' in key.lower():
                    print(f"  {key}: {value:.4f}")
                elif 'return' in key.lower() or 'drawdown' in key.lower():
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    if result.trade_history:
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {len(result.trade_history)}")

    if result.equity_curve is not None and len(result.equity_curve) > 0:
        print(f"\nEquity Curve:")
        print(f"  Start: {result.equity_curve.iloc[0]:,.2f}")
        print(f"  End: {result.equity_curve.iloc[-1]:,.2f}")
        total_return = (result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1)
        print(f"  Total Return: {total_return:.2%}")

    print(f"\n{'='*60}")


def _save_results(result, output_dir: str, prefix: str):
    """Save results to files."""
    import pandas as pd
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save equity curve
    if result.equity_curve is not None:
        equity_path = output_path / f"{prefix}_equity_curve.csv"
        result.equity_curve.to_csv(equity_path)
        print(f"Saved equity curve to: {equity_path}")

    # Save trades
    if result.trade_history:
        trades_path = output_path / f"{prefix}_trades.csv"
        pd.DataFrame(result.trade_history).to_csv(trades_path, index=False)
        print(f"Saved trades to: {trades_path}")

    # Save metrics
    if result.performance_metrics:
        metrics_path = output_path / f"{prefix}_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            # Convert non-serializable types
            metrics = {}
            for k, v in result.performance_metrics.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    metrics[k] = v
                else:
                    metrics[k] = str(v)
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")


if __name__ == '__main__':
    main()
