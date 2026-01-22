# CLAUDE.md - CTASectorTrendV2

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

CTASectorTrendV2 is a refactored version of CTASectorTrendV1 with key architectural improvements:

1. **Factor-Engine Decoupling**: Factors are completely independent of the execution engine
2. **4-Layer Configuration Hierarchy**: L1(Factor Type) → L2(Scope) → L3(Category) → L4(Sub-category)
3. **Common + Idiosyncratic Parallel Loading**: Both factor types must be computed and fused
4. **100% Logic Consistency**: Mathematical formulas unchanged, regression test with np.allclose(atol=1e-7)

## Key Commands

### Running the System

```bash
# Unified entry point (coming soon)
python run.py --mode pair_trading --factor copula

# Single-factor backtest
python run.py --mode single_factor --factor HurstExponent

# Multi-factor backtest
python run.py --mode multi_factor --combination comprehensive
```

### Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

### Directory Structure

```
CTASectorTrendV2/
├── Reports/reproduction/           # Specification docs and audit reports
│   ├── factor_specs/               # Factor specifications
│   └── audit_reports/              # Generated audit reports
│
├── config/
│   ├── multi_factor_config.yaml    # Main config (4-layer hierarchy)
│   ├── factor_registry.yaml        # Factor class mapping and metadata
│   └── backtest_config.yaml        # Backtest parameters
│
├── core/
│   ├── factors/                    # Factor implementations (decoupled)
│   │   ├── base.py                 # FactorBase ABC (unified calculate interface)
│   │   ├── registry.py             # FactorRegistry (dynamic instantiation)
│   │   ├── time_series/
│   │   │   ├── base.py             # TimeSeriesFactorBase
│   │   │   ├── liquidity.py        # AMIHUD, Amivest
│   │   │   └── trend.py            # HurstExponent, EMDTrend
│   │   ├── cross_sectional/
│   │   │   └── base.py             # CrossSectionalFactorBase
│   │   └── pair_trading/
│   │       ├── base.py             # PairTradingFactorBase
│   │       └── copula.py           # CopulaPairFactor
│   │
│   ├── engine/                     # Pure execution engine (no factor logic)
│   │   └── execution.py            # TradeExecutor, CapitalAllocator
│   │
│   ├── processors/                 # Signal processing and multi-layer fusion
│   │   ├── config_loader.py        # HierarchicalConfigLoader (V2 config)
│   │   ├── factor_engine.py        # FactorComputeEngine (orchestration only)
│   │   ├── normalizer.py           # SignalNormalizer (zscore_clip/minmax/rank)
│   │   ├── combiner.py             # MultiLayerCombiner (L1-L4 fusion)
│   │   └── param_resolver.py       # ParameterResolver (idio→sector→common)
│   │
│   ├── data/
│   │   └── loader.py               # DataLoader
│   │
│   └── metrics/
│       ├── performance.py          # PerformanceMetrics
│       └── data_logger.py          # DataLogger
│
├── tests/
│   ├── single_factor/              # Single factor unit tests
│   ├── multi_factor/               # Multi-factor integration tests
│   └── regression/                 # V1 vs V2 regression tests
│       ├── test_v1_v2_consistency.py
│       └── snapshots/              # V1 output snapshots
│
├── run.py                          # Unified entry point
└── CLAUDE.md                       # This file
```

### Core Interfaces

#### FactorBase (Unified Interface)

```python
class FactorBase(ABC):
    factor_type: FactorType = NotImplemented
    default_normalization: str = "zscore_clip"

    @abstractmethod
    def calculate(self, *args, **kwargs) -> Union[float, pd.Series, Dict]:
        """Core calculation method - replaces V1's compute()"""
        pass

    # Backward compatibility: compute() calls calculate()
    def compute(self, *args, **kwargs) -> Any:
        return self.calculate(*args, **kwargs)
```

#### Factor Type System

| Factor Type | Input | Output | Example |
|-------------|-------|--------|---------|
| `TIME_SERIES` | Single asset DataFrame | `float` in [-1, 1] | HurstExponent |
| `XS_PAIRWISE` | 2 asset DataFrames | `float` in [-1, 1] | CopulaPairFactor |
| `XS_GLOBAL` | All assets dict | `pd.Series` (ranks) | MomentumRank |

### 4-Layer Configuration Hierarchy

```
L1: Factor Type      │ time_series: 0.50, cross_sectional: 0.35, pair_trading: 0.15
        │
        ▼
L2: Application Scope│ Common (all assets) + Idiosyncratic (specific assets/sectors)
        │
        ▼
L3: Logical Category │ trend, volatility, liquidity, momentum, copula, etc.
        │
        ▼
L4: Sub-category     │ Specific factor implementations (HurstExponent, AMIHUD, etc.)
```

**Parameter Priority**: Asset > Sector > Common

### Data Flow Pipeline

```
1. LOAD CONFIG
   HierarchicalConfigLoader.load()
   ├── Parse L1 factor_type_weights
   ├── Parse L2 common/idiosyncratic
   ├── Parse L3 categories
   └── Parse L4 sub-categories

2. FETCH DATA
   DataLoader.load() → Dict[ticker, DataFrame]

3. CALCULATE FACTORS (Common + Idiosyncratic both computed!)
   For each asset:
   ├── ParameterResolver.get_applicable_factors() → [all applicable factors]
   ├── ParameterResolver.resolve_params() → merged params
   ├── FactorRegistry.get_or_create() → factor instance
   └── factor.calculate(data) → raw_value

4. SIGNAL LAYER
   SignalNormalizer.normalize() → [-1, 1]
   MultiLayerCombiner.combine() → final_signal

5. EXECUTE
   ExecutionEngine.run() → BacktestResult
```

### Multi-Layer Combination

```python
class MultiLayerCombiner:
    def combine(self, factor_outputs, asset=None) -> float:
        """
        4-layer fusion:
        L4: Equal weight within sub-category
        L3: Equal weight within category
        L2: Equal weight for Common + Idio
        L1: Weighted by factor_type_weights
        """
        l1_weights = {
            'time_series': 0.50,
            'cross_sectional': 0.35,
            'pair_trading': 0.15
        }
```

## Configuration Files

### multi_factor_config.yaml

Main configuration with 4-layer hierarchy:

```yaml
factor_type_weights:
  time_series: 0.50
  cross_sectional: 0.35
  pair_trading: 0.15

common:
  time_series:
    - name: HurstExponent
      class: core.factors.time_series.trend.HurstExponent
      category: trend
      default_params:
        window: 100

idiosyncratic:
  by_sector:
    Wind煤焦钢矿:
      time_series:
        - name: HurstExponent
          params:
            window: 120  # Override for this sector
```

## Testing

### Regression Testing (V1 vs V2)

```python
class V1V2RegressionTest:
    """Zero-modification policy verification"""

    def test_factor_values_match(self):
        """Factor values must match"""
        assert np.allclose(v1_vals, v2_vals, atol=1e-7)

    def test_signals_match(self):
        """Combined signals must match"""
        assert np.allclose(v1_sig, v2_sig, atol=1e-7)

    def test_equity_curve_match(self):
        """Equity curves must match"""
        assert np.allclose(v1_eq, v2_eq, atol=1e-7)
```

## Important Conventions

- **Unified Entry Point**: Use `python run.py` from project root
- **Factor Interface**: All factors implement `calculate()` (not `compute()`)
- **Parameter Resolution**: Always use `ParameterResolver` for hierarchical params
- **Signal Normalization**: Apply `SignalNormalizer` before combination
- **Backward Compatibility**: `compute()` exists as alias for `calculate()`

## Key Differences from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Factor method | `compute()` | `calculate()` (with `compute()` alias) |
| Config layers | 3 layers | 4 layers (L1-L4) |
| Factor loading | Direct | Via `FactorRegistry` |
| Parameter resolution | Manual | `ParameterResolver` with priority |
| Signal combination | Single mode | `MultiLayerCombiner` with 4-layer fusion |
| Idiosyncratic factors | Optional | MUST be computed with Common |

## Migration from V1

1. Change `factor.compute()` to `factor.calculate()` (or keep using `compute()`)
2. Update config to use 4-layer hierarchy
3. Ensure both Common AND Idiosyncratic factors are computed
4. Use `ParameterResolver` for parameter merging
5. Run regression tests to verify 100% logic consistency
