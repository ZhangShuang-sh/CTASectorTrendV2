#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
File Name:      execution
Description:    Trade execution and capital allocation module
Author:         CTASectorTrendV2
Date:           2026/01/13
-------------------------------------------------

Provides margin-based trade execution with 4 trade types:
- OPEN: New position from zero
- CLOSE: Position to zero
- FLIP: Reverse direction (close + open opposite)
- ADJUST: Same direction change (add/reduce)

Also provides capital allocation for multiple pairs:
- equal: Equal allocation across pairs
- weighted: Weighted by signal strength
- volatility_parity: Risk parity based on volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.factors.utils.math_utils import VolatilityCalculator, create_volatility_calculator

# =============================================================================
# DYNAMIC RISK MANAGEMENT PARAMETERS
# =============================================================================
# Fallback hard limit (only used if dynamic calculation fails)
FALLBACK_MAX_LOTS = 500       # Safety fallback - should rarely be hit
HARD_MIN_LOTS = 1             # Minimum trade unit
DEFAULT_ATR_WINDOW = 14       # ATR calculation window

# Default risk parameters for DYNAMIC position sizing
DEFAULT_MAX_ALLOCATION_PCT = 0.10    # 10% of capital per asset (max notional)
DEFAULT_LEVERAGE_SCALER = 2.0        # Boost factor for ATR-based sizing (2.0-3.0x)
DEFAULT_RISK_BUDGET_PCT = 0.02       # 2% risk budget per trade

# Legacy hard cap (kept as ultimate safety fallback)
HARD_MAX_LOTS_PER_ASSET = 500  # Absolute maximum - should never be hit with dynamic limits

# Contract multipliers for Chinese futures
# Key: Product code prefix, Value: Contract multiplier
CONTRACT_MULTIPLIERS = {
    # 黑色系 (Ferrous metals)
    'RB': 10,    # 螺纹钢 10吨/手
    'HC': 10,    # 热卷 10吨/手
    'I': 100,    # 铁矿石 100吨/手
    'J': 100,    # 焦炭 100吨/手
    'JM': 60,    # 焦煤 60吨/手
    'SF': 5,     # 硅铁 5吨/手
    'SM': 5,     # 锰硅 5吨/手
    'SS': 5,     # 不锈钢 5吨/手

    # 有色金属 (Non-ferrous metals)
    'CU': 5,     # 铜 5吨/手
    'AL': 5,     # 铝 5吨/手
    'ZN': 5,     # 锌 5吨/手
    'PB': 5,     # 铅 5吨/手
    'NI': 1,     # 镍 1吨/手
    'SN': 1,     # 锡 1吨/手

    # 贵金属 (Precious metals)
    'AU': 1000,  # 黄金 1000克/手
    'AG': 15,    # 白银 15千克/手

    # 能源化工 (Energy & Chemicals)
    'SC': 1000,  # 原油 1000桶/手
    'FU': 10,    # 燃油 10吨/手
    'BU': 10,    # 沥青 10吨/手
    'TA': 5,     # PTA 5吨/手
    'MA': 10,    # 甲醇 10吨/手
    'EG': 10,    # 乙二醇 10吨/手
    'PP': 5,     # 聚丙烯 5吨/手
    'L': 5,      # 塑料 5吨/手
    'V': 5,      # PVC 5吨/手
    'EB': 5,     # 苯乙烯 5吨/手
    'PG': 20,    # LPG 20吨/手
    'RU': 10,    # 橡胶 10吨/手
    'NR': 10,    # 20号胶 10吨/手
    'SP': 10,    # 纸浆 10吨/手

    # 农产品 (Agricultural products)
    'C': 10,     # 玉米 10吨/手
    'CS': 10,    # 淀粉 10吨/手
    'A': 10,     # 豆一 10吨/手
    'B': 10,     # 豆二 10吨/手
    'M': 10,     # 豆粕 10吨/手
    'Y': 10,     # 豆油 10吨/手
    'P': 10,     # 棕榈油 10吨/手
    'OI': 10,    # 菜油 10吨/手
    'RM': 10,    # 菜粕 10吨/手
    'SR': 10,    # 白糖 10吨/手
    'CF': 5,     # 棉花 5吨/手
    'AP': 10,    # 苹果 10吨/手
    'CJ': 5,     # 红枣 5吨/手
    'JD': 10,    # 鸡蛋 10吨/手 (actually 5 tons)
    'LH': 16,    # 生猪 16吨/手

    # 金融期货 (Financial futures)
    'IF': 300,   # 沪深300指数 300点/手
    'IC': 200,   # 中证500指数 200点/手
    'IH': 300,   # 上证50指数 300点/手
    'IM': 200,   # 中证1000指数 200点/手
    'T': 10000,  # 10年期国债 10000元/点
    'TF': 10000, # 5年期国债 10000元/点
    'TS': 20000, # 2年期国债 20000元/点
}

# Global trade counter for verification logging
_trade_counter = 0
_verification_enabled = True
_sample_log_printed = False


def get_contract_multiplier(symbol: str) -> int:
    """
    Get contract multiplier for a symbol.

    Args:
        symbol: Full symbol (e.g., 'RB.SHF') or product code (e.g., 'RB')

    Returns:
        Contract multiplier (defaults to 10 with warning if unknown)
    """
    # Extract product code from symbol
    if '.' in symbol:
        product_code = symbol.split('.')[0]
    else:
        product_code = symbol

    # Remove trailing digits (e.g., 'RB2401' -> 'RB')
    import re
    product_code = re.sub(r'\d+$', '', product_code.upper())

    multiplier = CONTRACT_MULTIPLIERS.get(product_code)
    if multiplier is None:
        print(f"[WARNING] Unknown contract multiplier for {symbol}, using default 10")
        return 10

    return multiplier


def reset_trade_verification():
    """Reset the trade verification counter for new test runs."""
    global _trade_counter, _sample_log_printed
    _trade_counter = 0
    _sample_log_printed = False
    print(f"[VERIFICATION] Trade counter reset. Dynamic risk management enabled.")
    print(f"  - Max allocation: {DEFAULT_MAX_ALLOCATION_PCT*100:.0f}% of capital per asset")
    print(f"  - Leverage scaler: {DEFAULT_LEVERAGE_SCALER}x")


def disable_trade_verification():
    """Disable verification logging (for production runs)."""
    global _verification_enabled
    _verification_enabled = False


def enable_trade_verification():
    """Enable verification logging."""
    global _verification_enabled
    _verification_enabled = True


@dataclass
class TradeRecord:
    """Trade record data structure"""
    date: pd.Timestamp
    symbol: str
    action: str  # OPEN, CLOSE, FLIP, ADJUST
    quantity: int
    price: float
    value: float
    cost: float
    pnl: float
    reason: str
    signal_info: Optional[str] = None
    vol_info: Optional[Dict] = None


class TradeExecutor:
    """
    Margin-based trade execution with proper accounting

    Handles 4 trade types:
    - OPEN: New position from zero
    - CLOSE: Position to zero
    - FLIP: Reverse direction
    - ADJUST: Same direction change

    Supports volatility parity for pair trading:
    - equal: Equal capital split between legs (50/50)
    - volatility_parity: Inverse volatility weighted (Position_A * Vol_A = Position_B * Vol_B)

    Attributes:
        transaction_cost: Transaction cost ratio (default 0.001 = 0.1%)
        slippage: Slippage ratio (default 0.0002 = 0.02%)
        margin_rate: Margin requirement ratio (default 0.1 = 10%)
        max_position_ratio: Max single position ratio (default 0.3 = 30%)
        use_volatility_parity: Enable volatility parity for pair sizing (default True)
        volatility_window: Window for volatility calculation (default 20)
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        slippage: float = 0.0002,
        margin_rate: float = 0.1,
        max_position_ratio: float = 0.3,
        use_volatility_parity: bool = True,
        volatility_window: int = 20,
        volatility_method: str = 'stddev',
        # Dynamic risk management parameters
        max_allocation_pct: float = DEFAULT_MAX_ALLOCATION_PCT,
        leverage_scaler: float = DEFAULT_LEVERAGE_SCALER,
        risk_budget_pct: float = DEFAULT_RISK_BUDGET_PCT
    ):
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.margin_rate = margin_rate
        self.max_position_ratio = max_position_ratio
        self.use_volatility_parity = use_volatility_parity
        self.volatility_window = volatility_window
        self.volatility_method = volatility_method
        self.trade_history: List[TradeRecord] = []

        # Dynamic risk management parameters
        self.max_allocation_pct = max_allocation_pct  # Max % of capital per asset
        self.leverage_scaler = leverage_scaler        # Position sizing multiplier
        self.risk_budget_pct = risk_budget_pct        # Risk budget per trade

        # Initialize volatility calculator
        self.volatility_calculator = create_volatility_calculator(
            window=volatility_window,
            method=volatility_method,
            annualize=True
        )

    def execute_trade(
        self,
        portfolio: Dict,
        symbol: str,
        target_quantity: int,
        date_prices: Dict[str, Dict],
        reason: str,
        signal: Any = None,
        vol_info: Dict = None
    ) -> Tuple[Dict, Optional[TradeRecord]]:
        """
        Execute trade with proper margin accounting

        CRITICAL: This method enforces HARD_MAX_LOTS_PER_ASSET on final positions.

        Parameters
        ----------
        portfolio : dict
            Current portfolio state with keys:
            - cash: Available cash
            - positions: {symbol: quantity}
            - entry_prices: {symbol: price}
            - date: Current date
        symbol : str
            Symbol to trade
        target_quantity : int
            Target position quantity (positive=long, negative=short, 0=close)
        date_prices : dict
            Price data {symbol: {'close': price, ...}}
        reason : str
            Trade reason for logging
        signal : any, optional
            Signal object for logging
        vol_info : dict, optional
            Volatility info for logging

        Returns
        -------
        portfolio : dict
            Updated portfolio state
        trade_record : TradeRecord or None
            Trade record if executed, None if skipped
        """
        if symbol not in date_prices:
            return portfolio, None

        current_price = date_prices[symbol].get('close')
        if current_price is None or current_price <= 0:
            return portfolio, None

        current_quantity = portfolio['positions'].get(symbol, 0)

        # =====================================================================
        # DYNAMIC POSITION LIMIT - Based on portfolio capital and max_allocation_pct
        # max_allowed_value = portfolio_capital * max_allocation_pct
        # dynamic_limit = max_allowed_value / (price * contract_multiplier)
        # =====================================================================
        if target_quantity != 0:  # Only cap non-zero targets (allow closing to 0)
            portfolio_capital = portfolio.get('total_value', portfolio.get('cash', 10_000_000))
            multiplier = get_contract_multiplier(symbol)
            max_allowed_value = portfolio_capital * self.max_allocation_pct
            notional_per_lot = current_price * multiplier

            # Calculate dynamic limit
            if notional_per_lot > 0:
                dynamic_limit = int(max_allowed_value / notional_per_lot)
                dynamic_limit = max(HARD_MIN_LOTS, dynamic_limit)  # At least 1 lot
            else:
                dynamic_limit = FALLBACK_MAX_LOTS

            # Apply dynamic limit (capped by FALLBACK_MAX_LOTS as ultimate safety)
            dynamic_limit = min(dynamic_limit, FALLBACK_MAX_LOTS)
            target_quantity = min(dynamic_limit, max(-dynamic_limit, target_quantity))

        # ========== FIX: Close on Zero Validation ==========
        # If target is 0 and current is already 0, there's nothing to close
        if target_quantity == 0 and current_quantity == 0:
            return portfolio, None  # Cannot close a position that doesn't exist

        # Determine trade type
        is_opening = (current_quantity == 0 and target_quantity != 0)
        is_closing = (target_quantity == 0 and current_quantity != 0)
        is_flipping = (current_quantity != 0 and target_quantity != 0 and
                       np.sign(current_quantity) != np.sign(target_quantity))
        is_adjusting = (current_quantity != 0 and target_quantity != 0 and
                        np.sign(current_quantity) == np.sign(target_quantity))

        # Calculate trade quantity
        trade_quantity = target_quantity - current_quantity

        if trade_quantity == 0:
            return portfolio, None

        # Apply slippage
        execution_price = current_price * (1 + np.sign(trade_quantity) * self.slippage)

        # Calculate trade value and cost
        trade_value = abs(trade_quantity) * execution_price
        txn_cost = trade_value * self.transaction_cost

        # Margin required
        margin_required = trade_value * self.margin_rate

        # Initialize variables
        pnl = 0.0
        action = ''

        if is_opening:
            # Opening new position
            required_cash = margin_required + txn_cost
            if portfolio['cash'] < required_cash:
                return portfolio, None  # Insufficient cash

            portfolio['cash'] -= required_cash
            portfolio['entry_prices'][symbol] = execution_price
            action = 'OPEN'
            pnl = 0

        elif is_closing:
            # Closing position
            entry_price = portfolio['entry_prices'].get(symbol, execution_price)
            # P&L = (exit_price - entry_price) * quantity
            pnl = (execution_price - entry_price) * current_quantity
            # Return original margin (based on entry price)
            old_margin = abs(current_quantity) * entry_price * self.margin_rate
            portfolio['cash'] += old_margin + pnl - txn_cost
            # Clear entry price
            portfolio['entry_prices'].pop(symbol, None)
            action = 'CLOSE'

        elif is_flipping:
            # Flip position: close then open opposite
            entry_price = portfolio['entry_prices'].get(symbol, execution_price)

            # 1. Close existing position
            close_pnl = (execution_price - entry_price) * current_quantity
            old_margin = abs(current_quantity) * entry_price * self.margin_rate
            close_cost = abs(current_quantity) * execution_price * self.transaction_cost
            portfolio['cash'] += old_margin + close_pnl - close_cost

            # 2. Open new position (opposite direction)
            new_margin = abs(target_quantity) * execution_price * self.margin_rate
            open_cost = abs(target_quantity) * execution_price * self.transaction_cost

            # Check if enough cash for new position
            if portfolio['cash'] < new_margin + open_cost:
                # Not enough cash, only complete the close
                portfolio['positions'][symbol] = 0
                portfolio['entry_prices'].pop(symbol, None)
                # Record close trade
                close_record = TradeRecord(
                    date=portfolio.get('date'),
                    symbol=symbol,
                    action='CLOSE',
                    quantity=-current_quantity,
                    price=execution_price,
                    value=abs(current_quantity) * execution_price,
                    cost=close_cost,
                    pnl=close_pnl,
                    reason=reason + '_翻仓失败_平仓',
                    signal_info=str(signal) if signal is not None else None,
                    vol_info=vol_info
                )
                self.trade_history.append(close_record)
                return portfolio, close_record

            portfolio['cash'] -= new_margin + open_cost
            portfolio['entry_prices'][symbol] = execution_price
            action = 'FLIP'
            pnl = close_pnl
            txn_cost = close_cost + open_cost

        else:  # is_adjusting
            # Same direction adjustment (add or reduce)
            entry_price = portfolio['entry_prices'].get(symbol, execution_price)
            old_margin = abs(current_quantity) * entry_price * self.margin_rate
            new_margin = abs(target_quantity) * execution_price * self.margin_rate

            if abs(target_quantity) > abs(current_quantity):
                # Adding to position
                additional_margin = new_margin - old_margin + txn_cost
                if portfolio['cash'] < additional_margin:
                    return portfolio, None  # Insufficient cash

                portfolio['cash'] -= additional_margin
                # Update weighted average cost
                added_quantity = abs(target_quantity) - abs(current_quantity)
                total_cost = entry_price * abs(current_quantity) + execution_price * added_quantity
                portfolio['entry_prices'][symbol] = total_cost / abs(target_quantity)
                pnl = 0  # No realized P&L on add
            else:
                # Reducing position
                closed_quantity = abs(current_quantity) - abs(target_quantity)
                # P&L = (current_price - entry_price) * closed_quantity * direction
                pnl = (execution_price - entry_price) * closed_quantity * np.sign(current_quantity)
                returned_margin = old_margin - new_margin
                portfolio['cash'] += returned_margin + pnl - txn_cost
                # Entry price remains unchanged

            action = 'ADJUST'

        # =====================================================================
        # FINAL SAFETY CHECK: Ensure the position being set is within dynamic limits
        # Recalculate dynamic limit as a final safety check
        # =====================================================================
        if target_quantity != 0:
            portfolio_capital = portfolio.get('total_value', portfolio.get('cash', 10_000_000))
            multiplier = get_contract_multiplier(symbol)
            max_allowed_value = portfolio_capital * self.max_allocation_pct
            notional_per_lot = execution_price * multiplier  # Use execution price

            if notional_per_lot > 0:
                dynamic_limit = int(max_allowed_value / notional_per_lot)
                dynamic_limit = max(HARD_MIN_LOTS, min(FALLBACK_MAX_LOTS, dynamic_limit))
            else:
                dynamic_limit = FALLBACK_MAX_LOTS

            final_position = min(dynamic_limit, max(-dynamic_limit, target_quantity))
            if final_position != target_quantity:
                print(f"[POSITION CAP] {symbol}: Capped {target_quantity} -> {final_position} (dynamic limit: {dynamic_limit})")
                target_quantity = final_position

        # Update position
        portfolio['positions'][symbol] = target_quantity

        # Create trade record
        trade_record = TradeRecord(
            date=portfolio.get('date'),
            symbol=symbol,
            action=action,
            quantity=trade_quantity,
            price=execution_price,
            value=trade_value,
            cost=txn_cost,
            pnl=pnl,
            reason=reason,
            signal_info=str(signal) if signal is not None else None,
            vol_info=vol_info
        )

        self.trade_history.append(trade_record)

        return portfolio, trade_record

    def calculate_position_quantities(
        self,
        portfolio: Dict,
        pair: Tuple[str, str],
        allocated_capital: float,
        date_prices: Dict[str, Dict],
        backtest_data: pd.DataFrame = None,
        current_date: pd.Timestamp = None
    ) -> Optional[Tuple[int, int]]:
        """
        Calculate position sizes for a pair trade with DYNAMIC CAPITAL-BASED LIMITS.

        Dynamic Risk Management Logic:
        1. Calculate max allowed notional per asset = portfolio_capital * max_allocation_pct
        2. Convert to quantity limit = max_notional / (price * contract_multiplier)
        3. Calculate ATR-based quantity with leverage_scaler boost
        4. Apply dynamic limit (not hard limit)

        Position sizing uses ATR-based volatility parity with leverage scaling:
        Position = (Risk_Budget / ATR) * leverage_scaler, capped by capital %

        Parameters
        ----------
        portfolio : dict
            Current portfolio state (must contain 'total_value' or 'cash')
        pair : tuple
            (symbol1, symbol2)
        allocated_capital : float
            Capital allocated to this pair (used for risk budget calculation)
        date_prices : dict
            Current prices
        backtest_data : DataFrame, optional
            Historical data for ATR calculation
        current_date : Timestamp, optional
            Current date for filtering

        Returns
        -------
        quantities : tuple or None
            (quantity1, quantity2) dynamically limited by capital percentage
        """
        global _trade_counter, _verification_enabled, _sample_log_printed

        symbol1, symbol2 = pair

        if symbol1 not in date_prices or symbol2 not in date_prices:
            return None

        price1 = date_prices[symbol1].get('close')
        price2 = date_prices[symbol2].get('close')

        if price1 is None or price2 is None or price1 <= 0 or price2 <= 0:
            return None

        # Get current portfolio capital for dynamic limit calculation
        portfolio_capital = portfolio.get('total_value', portfolio.get('cash', 10_000_000))

        # Get contract multipliers
        multiplier1 = get_contract_multiplier(symbol1)
        multiplier2 = get_contract_multiplier(symbol2)

        # =====================================================================
        # DYNAMIC POSITION LIMIT CALCULATION
        # max_allowed_value = portfolio_capital * max_allocation_pct
        # limit_quantity = max_allowed_value / (price * contract_multiplier)
        # =====================================================================
        max_allowed_value1 = portfolio_capital * self.max_allocation_pct
        max_allowed_value2 = portfolio_capital * self.max_allocation_pct

        dynamic_limit1 = int(max_allowed_value1 / (price1 * multiplier1))
        dynamic_limit2 = int(max_allowed_value2 / (price2 * multiplier2))

        # Ensure minimum of 1 lot
        dynamic_limit1 = max(HARD_MIN_LOTS, dynamic_limit1)
        dynamic_limit2 = max(HARD_MIN_LOTS, dynamic_limit2)

        # Calculate ATR-based quantities with leverage scaler
        raw_quantity1, raw_quantity2 = self._calculate_atr_based_quantities(
            pair=pair,
            price1=price1,
            price2=price2,
            allocated_capital=allocated_capital,
            backtest_data=backtest_data,
            current_date=current_date
        )

        # Apply dynamic limits (use min of ATR-based and capital-based limits)
        quantity1 = min(dynamic_limit1, raw_quantity1)
        quantity2 = min(dynamic_limit2, raw_quantity2)

        # Final safety cap (should rarely be hit with proper dynamic limits)
        quantity1 = min(FALLBACK_MAX_LOTS, max(HARD_MIN_LOTS, quantity1))
        quantity2 = min(FALLBACK_MAX_LOTS, max(HARD_MIN_LOTS, quantity2))

        # =====================================================================
        # VERIFICATION OUTPUT - Sample trade calculation
        # =====================================================================
        if _verification_enabled and not _sample_log_printed:
            _sample_log_printed = True
            print("\n" + "=" * 70)
            print("DYNAMIC RISK MANAGEMENT - Sample Trade Calculation")
            print("=" * 70)
            print(f"Pair: {symbol1} / {symbol2}")
            print(f"Portfolio Capital: {portfolio_capital:,.0f}")
            print(f"Max Allocation %: {self.max_allocation_pct * 100:.1f}%")
            print(f"Leverage Scaler: {self.leverage_scaler}x")
            print("-" * 70)
            print(f"Asset 1 ({symbol1}):")
            print(f"  Price: {price1:,.2f}")
            print(f"  Contract Multiplier: {multiplier1}")
            print(f"  Notional per Lot: {price1 * multiplier1:,.2f}")
            print(f"  Max Allowed Value: {max_allowed_value1:,.0f}")
            print(f"  Dynamic Limit: {dynamic_limit1} lots")
            print(f"  ATR-based Quantity: {raw_quantity1} lots")
            print(f"  Final Quantity: {quantity1} lots")
            print("-" * 70)
            print(f"Asset 2 ({symbol2}):")
            print(f"  Price: {price2:,.2f}")
            print(f"  Contract Multiplier: {multiplier2}")
            print(f"  Notional per Lot: {price2 * multiplier2:,.2f}")
            print(f"  Max Allowed Value: {max_allowed_value2:,.0f}")
            print(f"  Dynamic Limit: {dynamic_limit2} lots")
            print(f"  ATR-based Quantity: {raw_quantity2} lots")
            print(f"  Final Quantity: {quantity2} lots")
            print("=" * 70 + "\n")

        # Additional verification for first 5 trades
        if _verification_enabled and _trade_counter < 5:
            _trade_counter += 1
            print(f"[POSITION #{_trade_counter}] {pair}: Qty=({quantity1}, {quantity2}), "
                  f"DynamicLimits=({dynamic_limit1}, {dynamic_limit2}), "
                  f"RawATR=({raw_quantity1}, {raw_quantity2})")

        return quantity1, quantity2

    def _calculate_atr_based_quantities(
        self,
        pair: Tuple[str, str],
        price1: float,
        price2: float,
        allocated_capital: float,
        backtest_data: pd.DataFrame,
        current_date: pd.Timestamp = None
    ) -> Tuple[int, int]:
        """
        Calculate ATR-based volatility parity position sizes with leverage scaling.

        Uses ATR (Average True Range) for volatility measurement.
        Formula: Position = (Risk_Budget / ATR) * leverage_scaler

        The risk budget is allocated inversely proportional to ATR.
        Higher ATR = Lower position size (less volatile = larger position).
        Leverage scaler boosts the final position size (default 2.0x).

        Parameters
        ----------
        pair : tuple
            (symbol1, symbol2)
        price1, price2 : float
            Current prices
        allocated_capital : float
            Total capital for this pair
        backtest_data : DataFrame
            Historical data for ATR calculation
        current_date : Timestamp
            Current date

        Returns
        -------
        (quantity1, quantity2) : tuple
            ATR-adjusted quantities with leverage scaling (caller applies capital limits)
        """
        symbol1, symbol2 = pair

        # Calculate ATR for both assets
        atr1 = self._calculate_atr(symbol1, backtest_data, current_date)
        atr2 = self._calculate_atr(symbol2, backtest_data, current_date)

        # Default to 2% of price if ATR calculation fails
        if atr1 <= 0:
            atr1 = price1 * 0.02
        if atr2 <= 0:
            atr2 = price2 * 0.02

        # Calculate inverse ATR weights (volatility parity)
        # Lower ATR = higher weight (more stable asset gets larger position)
        inv_atr1 = 1.0 / atr1
        inv_atr2 = 1.0 / atr2
        total_inv_atr = inv_atr1 + inv_atr2

        weight1 = inv_atr1 / total_inv_atr
        weight2 = inv_atr2 / total_inv_atr

        # Allocate risk budget using instance parameter
        total_risk_budget = allocated_capital * self.risk_budget_pct

        risk_budget1 = total_risk_budget * weight1
        risk_budget2 = total_risk_budget * weight2

        # Position = (Risk_Budget / ATR) * leverage_scaler
        # Base quantity from ATR-based risk budgeting
        base_quantity1 = risk_budget1 / atr1
        base_quantity2 = risk_budget2 / atr2

        # Apply leverage scaler to boost position sizes
        quantity1 = int(base_quantity1 * self.leverage_scaler)
        quantity2 = int(base_quantity2 * self.leverage_scaler)

        # Ensure minimum
        quantity1 = max(HARD_MIN_LOTS, quantity1)
        quantity2 = max(HARD_MIN_LOTS, quantity2)

        return quantity1, quantity2

    def _calculate_atr(
        self,
        symbol: str,
        backtest_data: pd.DataFrame,
        current_date: pd.Timestamp = None,
        window: int = DEFAULT_ATR_WINDOW
    ) -> float:
        """
        Calculate Average True Range (ATR) for an asset.

        ATR = Average of True Range over the lookback window.
        True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)

        Parameters
        ----------
        symbol : str
            Product code (e.g., 'RB', 'HC')
        backtest_data : DataFrame
            Historical data with TRADE_DT, PRODUCT_CODE, S_DQ_CLOSE, S_DQ_HIGH, S_DQ_LOW
        current_date : Timestamp
            Current date for filtering
        window : int
            ATR calculation window (default 14)

        Returns
        -------
        atr : float
            Average True Range (in price units)
        """
        if backtest_data is None:
            return 0.0

        try:
            # Filter data for this symbol
            symbol_data = backtest_data[backtest_data['PRODUCT_CODE'] == symbol].copy()

            if symbol_data.empty:
                return 0.0

            # Filter by date
            if current_date is not None:
                symbol_data = symbol_data[symbol_data['TRADE_DT'] <= current_date]

            if len(symbol_data) < window + 1:
                return 0.0

            # Sort and take recent data
            symbol_data = symbol_data.sort_values('TRADE_DT').tail(window + 10)

            # Get price columns
            close = symbol_data['S_DQ_CLOSE'].values
            high = symbol_data.get('S_DQ_HIGH', symbol_data['S_DQ_CLOSE']).values
            low = symbol_data.get('S_DQ_LOW', symbol_data['S_DQ_CLOSE']).values

            # Calculate True Range
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]

            tr1 = high - low  # High - Low
            tr2 = np.abs(high - prev_close)  # |High - PrevClose|
            tr3 = np.abs(low - prev_close)  # |Low - PrevClose|

            true_range = np.maximum(np.maximum(tr1, tr2), tr3)

            # ATR = Simple moving average of True Range
            atr = np.mean(true_range[-window:])

            return float(atr) if atr > 0 else 0.0

        except Exception as e:
            print(f"ATR calculation error for {symbol}: {e}")
            return 0.0

    def _calculate_volatility_parity_quantities(
        self,
        pair: Tuple[str, str],
        allocated_capital: float,
        price1: float,
        price2: float,
        backtest_data: pd.DataFrame,
        current_date: pd.Timestamp = None
    ) -> Tuple[int, int]:
        """
        DEPRECATED: Use calculate_position_quantities instead.

        This method is kept for backwards compatibility but now
        delegates to ATR-based calculation with dynamic limits.
        """
        qty1, qty2 = self._calculate_atr_based_quantities(
            pair=pair,
            price1=price1,
            price2=price2,
            allocated_capital=allocated_capital,
            backtest_data=backtest_data,
            current_date=current_date
        )

        # Safety fallback cap
        qty1 = min(FALLBACK_MAX_LOTS, max(HARD_MIN_LOTS, qty1))
        qty2 = min(FALLBACK_MAX_LOTS, max(HARD_MIN_LOTS, qty2))

        return qty1, qty2

    def _get_asset_volatility(
        self,
        symbol: str,
        backtest_data: pd.DataFrame,
        current_date: pd.Timestamp = None
    ) -> float:
        """
        Get volatility for a single asset

        Parameters
        ----------
        symbol : str
            Product code (e.g., 'RB', 'HC')
        backtest_data : DataFrame
            Historical data
        current_date : Timestamp, optional
            Current date for filtering

        Returns
        -------
        volatility : float
            Annualized volatility
        """
        if backtest_data is None:
            return self.volatility_calculator.default_volatility

        try:
            # Filter data for this symbol
            symbol_data = backtest_data[backtest_data['PRODUCT_CODE'] == symbol]

            if symbol_data.empty:
                return self.volatility_calculator.default_volatility

            # Filter by date if provided
            if current_date is not None:
                symbol_data = symbol_data[symbol_data['TRADE_DT'] <= current_date]

            if len(symbol_data) < self.volatility_calculator.min_periods:
                return self.volatility_calculator.default_volatility

            # Sort and take recent data
            symbol_data = symbol_data.sort_values('TRADE_DT').tail(self.volatility_window * 2)

            # Calculate volatility
            return self.volatility_calculator.calculate_volatility_from_dataframe(
                df=symbol_data,
                date_col='TRADE_DT',
                price_col='S_DQ_CLOSE',
                high_col='S_DQ_HIGH',
                low_col='S_DQ_LOW',
                current_date=current_date
            )

        except Exception:
            return self.volatility_calculator.default_volatility

    def get_trade_history(self) -> List[Dict]:
        """Get trade history as list of dicts"""
        return [
            {
                'date': t.date,
                'symbol': t.symbol,
                'action': t.action,
                'quantity': t.quantity,
                'price': t.price,
                'value': t.value,
                'cost': t.cost,
                'pnl': t.pnl,
                'reason': t.reason,
                'signal_info': t.signal_info,
                'vol_info': t.vol_info
            }
            for t in self.trade_history
        ]

    def clear_history(self):
        """Clear trade history"""
        self.trade_history = []


class CapitalAllocator:
    """
    Multi-pair capital allocation with risk parity support

    Supports 3 allocation methods:
    - equal: Equal allocation across all pairs
    - weighted: Weighted by signal strength
    - volatility_parity: Inverse volatility weighting (risk parity)

    The volatility_parity method implements true Risk Parity:
    Each pair receives capital inversely proportional to its volatility,
    ensuring equal risk contribution from each pair.

    Attributes:
        method: Allocation method ('equal', 'weighted', 'volatility_parity')
        max_position_ratio: Maximum position ratio for total budget
        default_volatility: Default volatility when calculation fails
        volatility_window: Window for volatility calculation (default 20)
    """

    def __init__(
        self,
        method: str = 'volatility_parity',
        max_position_ratio: float = 0.3,
        default_volatility: float = 0.15,
        volatility_window: int = 20,
        volatility_method: str = 'stddev'
    ):
        self.method = method
        self.max_position_ratio = max_position_ratio
        self.default_volatility = default_volatility
        self.volatility_window = volatility_window

        # Initialize shared volatility calculator
        self.volatility_calculator = create_volatility_calculator(
            window=volatility_window,
            method=volatility_method,
            annualize=True
        )

    def allocate(
        self,
        portfolio: Dict,
        entry_signals: List[Tuple],
        date_prices: Dict[str, Dict],
        backtest_data: pd.DataFrame = None,
        current_date: pd.Timestamp = None
    ) -> List[float]:
        """
        Allocate capital for multiple entry signals

        Parameters
        ----------
        portfolio : dict
            Current portfolio state
        entry_signals : list
            List of (key, signal) tuples for entries
        date_prices : dict
            Current prices
        backtest_data : DataFrame, optional
            Full historical data for volatility calculation
        current_date : Timestamp, optional
            Current date for filtering historical data

        Returns
        -------
        allocations : list
            Capital allocation for each signal
        """
        if not entry_signals:
            return []

        n_pairs = len(entry_signals)
        available_cash = portfolio.get('cash', 0)

        # Total margin budget = cash * max_position_ratio
        total_margin_budget = available_cash * self.max_position_ratio

        if self.method == 'equal':
            # Equal allocation
            allocation_per_pair = total_margin_budget / n_pairs
            allocations = [allocation_per_pair] * n_pairs

        elif self.method == 'weighted':
            # Weighted by signal strength
            strengths = []
            for key, signal in entry_signals:
                strength = getattr(signal, 'strength', 1.0) if hasattr(signal, 'strength') else 1.0
                if isinstance(signal, dict):
                    strength = signal.get('position_strength', 1.0)
                strengths.append(abs(strength))

            total_strength = sum(strengths)
            if total_strength <= 0:
                allocation_per_pair = total_margin_budget / n_pairs
                allocations = [allocation_per_pair] * n_pairs
            else:
                allocations = [
                    total_margin_budget * (s / total_strength)
                    for s in strengths
                ]

        elif self.method == 'volatility_parity':
            # Risk parity: inverse volatility weighting
            volatilities = self._calculate_pair_volatilities(
                entry_signals, backtest_data, current_date
            )

            if not volatilities or all(v <= 0 for v in volatilities):
                allocation_per_pair = total_margin_budget / n_pairs
                allocations = [allocation_per_pair] * n_pairs
            else:
                # Inverse volatility weights
                inv_vols = []
                for vol in volatilities:
                    if vol > 0:
                        inv_vols.append(1.0 / vol)
                    else:
                        inv_vols.append(1.0 / self.default_volatility)

                total_inv_vol = sum(inv_vols)
                allocations = [
                    total_margin_budget * (iv / total_inv_vol)
                    for iv in inv_vols
                ]
        else:
            # Default to equal
            allocation_per_pair = total_margin_budget / n_pairs
            allocations = [allocation_per_pair] * n_pairs

        return allocations

    def _calculate_pair_volatilities(
        self,
        entry_signals: List[Tuple],
        backtest_data: pd.DataFrame,
        current_date: pd.Timestamp,
        lookback: int = 60
    ) -> List[float]:
        """
        Calculate volatility for each pair

        Returns list of annualized volatilities
        """
        volatilities = []

        for key, signal in entry_signals:
            # Extract pair from signal
            pair = None
            if hasattr(signal, 'pair'):
                pair = signal.pair
            elif isinstance(signal, dict):
                pair = signal.get('pair')

            if pair is None:
                volatilities.append(self.default_volatility)
                continue

            vol = self._get_pair_volatility(pair, backtest_data, current_date, lookback)
            volatilities.append(vol)

        return volatilities

    def _get_pair_volatility(
        self,
        pair: Tuple[str, str],
        backtest_data: pd.DataFrame,
        current_date: pd.Timestamp,
        lookback: int = None
    ) -> float:
        """
        Calculate single pair's historical volatility

        Uses price ratio volatility for pair-level risk measurement.

        Parameters
        ----------
        pair : tuple
            (product1, product2)
        backtest_data : DataFrame
            Historical data with TRADE_DT, PRODUCT_CODE, S_DQ_CLOSE columns
        current_date : Timestamp
            Current date for filtering
        lookback : int, optional
            Lookback window in days (defaults to self.volatility_window)

        Returns
        -------
        volatility : float
            Annualized volatility
        """
        if lookback is None:
            lookback = self.volatility_window

        if backtest_data is None or current_date is None:
            return self.volatility_calculator.default_volatility

        try:
            # Filter historical data
            hist_data = backtest_data[
                (backtest_data['TRADE_DT'] <= current_date) &
                (backtest_data['PRODUCT_CODE'].isin(pair))
            ]

            if hist_data.empty:
                return self.volatility_calculator.default_volatility

            # Pivot: dates as index, products as columns
            pivot = hist_data.pivot_table(
                index='TRADE_DT',
                columns='PRODUCT_CODE',
                values='S_DQ_CLOSE'
            ).dropna()

            if len(pivot) < self.volatility_calculator.min_periods:
                return self.volatility_calculator.default_volatility

            if pair[0] not in pivot.columns or pair[1] not in pivot.columns:
                return self.volatility_calculator.default_volatility

            # Take last lookback days
            pivot = pivot.tail(lookback)

            # Calculate price ratio log returns
            price_ratio = pivot[pair[0]] / pivot[pair[1]]

            # Use the volatility calculator for consistent calculation
            vol = self.volatility_calculator.calculate_asset_volatility(price_ratio)

            return vol if vol > 0 else self.volatility_calculator.default_volatility

        except Exception:
            return self.volatility_calculator.default_volatility

    def _get_single_asset_volatility(
        self,
        symbol: str,
        backtest_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> float:
        """
        Calculate single asset's historical volatility

        Used for multi-asset risk parity allocation.

        Parameters
        ----------
        symbol : str
            Product code (e.g., 'RB', 'HC')
        backtest_data : DataFrame
            Historical data
        current_date : Timestamp
            Current date for filtering

        Returns
        -------
        volatility : float
            Annualized volatility
        """
        if backtest_data is None or current_date is None:
            return self.volatility_calculator.default_volatility

        try:
            # Filter data for this symbol
            symbol_data = backtest_data[backtest_data['PRODUCT_CODE'] == symbol]

            if symbol_data.empty:
                return self.volatility_calculator.default_volatility

            # Filter by date
            symbol_data = symbol_data[symbol_data['TRADE_DT'] <= current_date]

            if len(symbol_data) < self.volatility_calculator.min_periods:
                return self.volatility_calculator.default_volatility

            # Sort and take recent data
            symbol_data = symbol_data.sort_values('TRADE_DT').tail(self.volatility_window * 2)

            # Calculate volatility using shared calculator
            return self.volatility_calculator.calculate_volatility_from_dataframe(
                df=symbol_data,
                date_col='TRADE_DT',
                price_col='S_DQ_CLOSE',
                high_col='S_DQ_HIGH',
                low_col='S_DQ_LOW',
                current_date=current_date
            )

        except Exception:
            return self.volatility_calculator.default_volatility


class PortfolioUpdater:
    """
    Portfolio value updater with proper margin accounting

    Total Value = Cash + Margin Used + Unrealized P&L
    """

    def __init__(self, margin_rate: float = 0.1):
        self.margin_rate = margin_rate

    def update_portfolio_value(
        self,
        portfolio: Dict,
        date_prices: Dict[str, Dict]
    ) -> Dict:
        """
        Update portfolio value with margin accounting

        Parameters
        ----------
        portfolio : dict
            Portfolio state
        date_prices : dict
            Current prices

        Returns
        -------
        portfolio : dict
            Updated portfolio with:
            - total_value: Cash + margin_used + unrealized_pnl
            - margin_used: Total margin locked in
            - unrealized_pnl: Floating P&L
            - net_exposure: Net position value (signed)
            - gross_exposure: Total absolute position value
            - leverage: gross_exposure / total_value
        """
        margin_used = 0.0
        unrealized_pnl = 0.0
        net_exposure = 0.0
        gross_exposure = 0.0

        for symbol, quantity in portfolio.get('positions', {}).items():
            if quantity == 0:
                continue

            if symbol not in date_prices:
                continue

            current_price = date_prices[symbol].get('close')
            if current_price is None or current_price <= 0:
                continue

            entry_price = portfolio.get('entry_prices', {}).get(symbol, current_price)

            # Margin based on entry price (locked-in)
            margin_used += abs(quantity) * entry_price * self.margin_rate

            # Floating P&L
            unrealized_pnl += (current_price - entry_price) * quantity

            # Exposure calculations
            position_value = quantity * current_price
            net_exposure += position_value
            gross_exposure += abs(position_value)

        # Update portfolio
        portfolio['margin_used'] = margin_used
        portfolio['unrealized_pnl'] = unrealized_pnl
        portfolio['total_value'] = portfolio.get('cash', 0) + margin_used + unrealized_pnl
        portfolio['net_exposure'] = net_exposure
        portfolio['gross_exposure'] = gross_exposure

        # Leverage = gross exposure / total value
        if portfolio['total_value'] > 0:
            portfolio['leverage'] = gross_exposure / portfolio['total_value']
        else:
            portfolio['leverage'] = 0.0

        return portfolio


def filter_valid_entry_signals(
    entry_signals: List[Tuple],
    active_pairs: Dict
) -> List[Tuple]:
    """
    Filter out conflicting pair signals

    A signal is invalid if:
    - The pair is already active
    - Either product in the pair is already used in another active pair

    Parameters
    ----------
    entry_signals : list
        List of (key, signal) tuples
    active_pairs : dict
        Currently active pairs {pair: info}

    Returns
    -------
    valid_signals : list
        Filtered list of valid signals
    """
    valid = []
    used_products = set()

    # Collect products from active pairs
    for pair in active_pairs.keys():
        if isinstance(pair, tuple) and len(pair) >= 2:
            used_products.add(pair[0])
            used_products.add(pair[1])

    for key, signal in entry_signals:
        pair = None
        if hasattr(signal, 'pair'):
            pair = signal.pair
        elif isinstance(signal, dict):
            pair = signal.get('pair')

        if pair is None:
            valid.append((key, signal))
            continue

        # Check pair not already active
        if pair in active_pairs or (len(pair) >= 2 and tuple(reversed(pair)) in active_pairs):
            continue

        # Check products not occupied
        if pair[0] in used_products or pair[1] in used_products:
            continue

        valid.append((key, signal))
        used_products.add(pair[0])
        used_products.add(pair[1])

    return valid
