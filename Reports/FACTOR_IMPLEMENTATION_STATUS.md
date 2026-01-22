# 因子实现状态追踪

> 本文档追踪 `Reports/library/` 中研究报告与因子代码实现的对应关系。
>
> **最后更新**: 2026-01-22

---

## 统计概览

| 类别 | 报告总数 | 已实现 | 未实现 |
|------|---------|--------|--------|
| 时序因子 | 9 | 9 | 0 |
| 截面因子 | 3 | 3 | 0 |
| 配对因子 | 3 | 3 | 0 |
| **合计** | **15** | **15** | **0** |

---

## 已实现报告

### 时序因子 (`Reports/library/时序因子/`)

| 报告名称 | 对应实现文件 | 实现时间 |
|---------|-------------|---------|
| AMIHUD因子.docx | `core/factors/time_series/liquidity.py` | 2026-01-20 |
| Amivest因子.docx | `core/factors/time_series/liquidity.py` | 2026-01-20 |
| DUVOL因子.docx | `core/factors/time_series/volatility_advanced.py` | 2026-01-20 |
| 随机因子.docx | `core/factors/time_series/volatility_advanced.py` | 2026-01-20 |
| CTA研究系列之十三：基于统计语言模型（SLM）的择时交易研究.pdf | `core/factors/time_series/slm_timing.py` | 2026-01-20 |
| CTA研究系列之十九：带反转的加强版EMDT交易策略.pdf | `core/factors/time_series/trend.py` | 2026-01-20 |
| CTA研究系列之二十五：价量模式匹配股指期货交易策略.pdf | `core/factors/time_series/price_volume_pattern.py` | 2026-01-20 |
| CTA研究系列之二十七：均线交叉策略的另类创新研究.pdf | `core/factors/time_series/ma_crossover.py` | 2026-01-20 |
| 【中信期货金融工程】期货多因子系列之九：基于投资者行为的趋势因子——专题报告20241024.pdf | `core/factors/time_series/investor_behavior.py` | 2026-01-20 |

### 截面因子 (`Reports/library/截面因子/`)

| 报告名称 | 对应实现文件 | 实现时间 |
|---------|-------------|---------|
| VCRR因子.docx | `core/factors/cross_sectional/vcrr.py` | 2026-01-20 |
| 【中信期货金融工程】CTA风格因子手册（一）：量价类因子——专题报告20241016.pdf | `core/factors/cross_sectional/price_volume.py` | 2026-01-20 |
| 【中信期货金融工程】CTA风格因子手册（二）：动量类因子——专题报告20241106.pdf | `core/factors/cross_sectional/momentum_citic.py` | 2026-01-20 |

### 配对因子 (`Reports/library/配对因子/`)

| 报告名称 | 对应实现文件 | 实现时间 |
|---------|-------------|---------|
| 【中信期货金融工程】配对交易专题（三）：使用Copula函数套利的发散性思考——专题报告20240714.pdf | `core/factors/pair_trading/copula.py` | 2026-01-20 |
| 【中信期货金融工程】配对交易专题（二）：卡尔曼滤波在价差套利中的应用（基于Backtrader回测框架）——专题报告20240129.pdf | `core/factors/pair_trading/kalman_filter.py` | 2026-01-20 |
| CTA研究系列之二十二：风格动量下的股指期货跨品种套利策略.pdf | `core/factors/pair_trading/style_momentum.py` | 2026-01-20 |

---

## 未实现报告

> 当前所有报告均已实现。

*(暂无)*

---

## 报告-因子详细映射

### 时序因子详细映射

| 报告 | 实现因子类 |
|-----|-----------|
| AMIHUD因子.docx | `AMIHUDFactor` |
| Amivest因子.docx | `AmivestFactor` |
| DUVOL因子.docx | `DUVOLFactor` |
| 随机因子.docx | `RunsTestFactor` |
| CTA研究系列之十三 (SLM) | `SLMTimingFactor` |
| CTA研究系列之十九 (EMDT) | `EMDTrend` |
| CTA研究系列之二十五 (价量模式) | `PriceVolumePatternFactor` |
| CTA研究系列之二十七 (均线交叉) | `MACrossoverInnovationFactor` |
| 中信期货-投资者行为趋势因子 | `InvestorBehaviorTrendFactor`, `InvestorBehaviorMomentum`, `InvestorBehaviorReversal` |

### 截面因子详细映射

| 报告 | 实现因子类 |
|-----|-----------|
| VCRR因子.docx | `VCRRFactor`, `VCRRTimeSeriesFactor` |
| CTA风格因子手册（一）量价类 | `VolatilityFactor`, `CVFactor`, `SkewnessFactor`, `KurtosisFactor`, `AmplitudeFactor`, `LiquidityFactor` |
| CTA风格因子手册（二）动量类 | `CrossSectionalMomentumFactor`, `TimeSeriesMomentumFactor`, `CompositeMomentumFactor`, `BIASFactor`, `TrendStrengthFactor` |

### 配对因子详细映射

| 报告 | 实现因子类 |
|-----|-----------|
| 配对交易专题（三）Copula | `CopulaPairFactor` |
| 配对交易专题（二）卡尔曼滤波 | `KalmanFilterPairFactor` |
| CTA研究系列之二十二 (风格动量) | `StyleMomentumPairFactor`, `EnhancedStyleMomentumPairFactor` |

---

## 无报告来源的因子

以下因子基于经典方法实现，无特定研究报告：

| 因子名 | 实现文件 | 说明 |
|-------|---------|------|
| `HurstExponent` | `core/factors/time_series/trend.py` | 经典统计学方法 (Hurst, 1951) |
| `KalmanFilterDeviation` | `core/factors/time_series/volatility.py` | 通用卡尔曼滤波方法 |
| `KalmanTrendFollower` | `core/factors/time_series/volatility.py` | 通用卡尔曼滤波方法 |
| `MomentumRank` | `core/factors/cross_sectional/momentum.py` | 经典截面动量因子 |
| `TermStructure` | `core/factors/cross_sectional/momentum.py` | 期限结构经典因子 |
| `MemberHoldings` | `core/factors/cross_sectional/fundamental.py` | 基于交易所公开数据 |
| `TrendFollowingPairFactor` | `core/factors/pair_trading/trend_following.py` | 通用趋势跟踪方法 |

---

## 更新日志

| 日期 | 更新内容 |
|-----|---------|
| 2026-01-22 | 初始创建，完成15份报告的实现状态追踪 |
| 2026-01-20 | 完成全部34个因子的V2重构 |

---

## 维护说明

当新增报告或实现新因子时，请更新本文档：

1. **新增报告**: 添加到「未实现报告」部分
2. **完成实现**: 从「未实现」移至「已实现」，填写实现文件和时间
3. **更新统计**: 修改「统计概览」表格
4. **同步配置**: 确保 `config/factor_registry.yaml` 中的 `source_report` 字段同步更新
