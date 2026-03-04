<p align="center">
<a href="https://kavout.com">
<img src="https://i.imgur.com/DDetr8I.png" width="25%">
</a>
</p>

<h1 align="center">Zipline Refresh</h1>

<p align="center">
<strong>A high-performance Pythonic backtesting engine for algorithmic trading strategies</strong>
</p>

<p align="center">
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
<a href="https://pypi.python.org/pypi/zipline-refresh"><img src="https://img.shields.io/pypi/v/zipline-refresh" alt="PyPI"></a>
<a href="https://github.com/teleclaws/zipline-refresh/actions"><img src="https://img.shields.io/badge/tests-passing-brightgreen" alt="Tests"></a>
<a href="https://github.com/teleclaws/zipline-refresh/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
</p>

---

Zipline is a Pythonic event-driven system for backtesting, originally developed by [Quantopian](https://www.bizjournals.com/boston/news/2020/11/10/quantopian-shuts-down-cofounders-head-elsewhere.html). This **Refresh** fork modernizes the storage layer, eliminates legacy dependencies, and delivers significant performance improvements.

<p align="center">
<a href="https://kavout.com/academy/zipline-refresh"><strong>Documentation</strong></a> &nbsp;&middot;&nbsp;
<a href="https://pypi.org/project/zipline-refresh/"><strong>PyPI</strong></a> &nbsp;&middot;&nbsp;
<a href="https://labs.kavout.com"><strong>Website</strong></a> &nbsp;&middot;&nbsp;
<a href="https://github.com/teleclaws/zipline-refresh/issues"><strong>Report Bug</strong></a>
</p>

---

## What's New in Refresh

### Phase 1: bcolz &rarr; Apache Parquet

The legacy bcolz storage layer has been **fully replaced** with Apache Parquet via PyArrow:

| | bcolz (legacy) | Parquet (new) |
|---|---|---|
| **Format** | Custom binary + Cython | Standard columnar, zstd compressed |
| **Daily bars** | One ctable per field | Single `.parquet` file per bundle |
| **Minute bars** | Fixed-stride padding + Cython position math | Actual trading minutes only &mdash; no padding |
| **Dependencies** | bcolz (unmaintained, build failures on Python 3.12+) | pyarrow (actively maintained) |
| **Data types** | uint32 (lossy for prices > $42,949) | float64 (full precision) |
| **Interoperability** | Proprietary format | Standard Parquet &mdash; readable by pandas, Spark, DuckDB |
| **Compression** | None / blosc | zstd (2-5x smaller on disk) |
| **Early close handling** | Complex Cython exclusion logic | Eliminated &mdash; only real trading minutes stored |

### Phase 2: Profiling-Driven Hot Path Optimization

Systematic profiling (50 assets, 780 bars/session) identified and eliminated bottlenecks across the entire data layer:

| Optimization | Speedup | Detail |
|---|---|---|
| **bcolz &rarr; Parquet migration** | **N/A** | Eliminated unmaintained dependency, Cython position math, uint32 truncation |
| **Lazy per-field loading** | **3.2x** single field | Load only requested OHLCV fields instead of all 5 at once |
| **Vectorized lifetimes** | **5x** | Replace per-sid Python loop with single `pd.DataFrame` construction |
| **Batch resample aggregation** | **5x** | Batch `load_raw_arrays` in DailyHistoryAggregator instead of per-field calls |
| **NumPy int64 searchsorted** | **40x** per lookup | Replace `DatetimeIndex.get_loc()` (~4.3&micro;s) with `np.searchsorted` on int64 (~0.1&micro;s) |
| **Vectorized last-traded** | **17x** | `np.flatnonzero` on volume array instead of Python backward scan |

**Net result:** pandas DatetimeIndex overhead reduced from **46% &rarr; 6.5%** of hot-path time. Per-bar latency **0.6ms &rarr; 0.3ms**.

<details>
<summary><strong>Benchmark details (50 assets x 780 bars)</strong></summary>

```
Before (bcolz baseline → initial Parquet):
  pandas DatetimeIndex             46.0%  ██████████████████████████████████████████████
  get_value (reader)               13.0%  █████████████
  memoize/lazyval                  10.0%  ██████████

After (fully optimized Parquet):
  pandas DatetimeIndex              6.5%  ██████
  get_value (reader)               26.7%  ██████████████████████████
  memoize/lazyval                  12.9%  ████████████
  numpy operations                 12.2%  ████████████

Total hot-path time: 0.44s → 0.24s (1.8x faster)
Per-bar latency: 0.6ms → 0.3ms
```

Micro-benchmarks (500 sids x 1000 days):
- Single field load: 65.5ms → 20.6ms (3.2x)
- get_last_traded_dt: 3.4ms → 0.2ms (17x)
- _lifetimes_map: 5.5ms → 1.1ms (5x)
- Sequential get_value: 68.5ms → 23.1ms (3.0x)

</details>

---

## Features

- **Event-Driven Architecture** &mdash; Realistic simulation with proper order lifecycle, slippage, and commission models
- **Pipeline API** &mdash; Factor-based screening with 20+ built-in technical factors (RSI, MACD, Bollinger, Ichimoku, etc.) and easy `CustomFactor` extensibility
- **Factor Composition** &mdash; `rank()`, `zscore()`, `demean()`, `winsorize()`, `top(N)` with `groupby` for sector-neutral strategies
- **PyData Integration** &mdash; pandas DataFrames in/out, compatible with matplotlib, scipy, statsmodels, scikit-learn
- **Multi-Country Support** &mdash; 42 country domains with proper trading calendars via `exchange_calendars`
- **Minute & Daily Resolution** &mdash; Full minute-level backtesting with proper market open/close handling

## Installation

Zipline supports Python >= 3.10 and is compatible with current versions of [NumFOCUS](https://numfocus.org/sponsored-projects?_sft_project_category=python-interface) libraries.

### Using `pip`

```bash
pip install zipline-refresh
```

### From source

```bash
git clone https://github.com/teleclaws/zipline-refresh.git
cd zipline-refresh
pip install -e .
```

See the [documentation](https://kavout.com/academy/zipline-refresh) for detailed instructions.

## Quickstart

### Example 1: RSI Long/Short Pipeline Strategy

Use the Pipeline API to rank stocks by RSI and build a long/short portfolio — rebalanced daily:

```python
from zipline.api import attach_pipeline, order_target_percent, pipeline_output, schedule_function
from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import RSI


def make_pipeline():
    rsi = RSI()
    return Pipeline(
        columns={"longs": rsi.top(3), "shorts": rsi.bottom(3)},
    )


def initialize(context):
    attach_pipeline(make_pipeline(), "my_pipeline")
    schedule_function(rebalance)
    context.set_commission(commission.PerShare(cost=0.001, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())


def before_trading_start(context, data):
    context.pipeline_data = pipeline_output("my_pipeline")


def rebalance(context, data):
    pipeline_data = context.pipeline_data
    longs = pipeline_data.index[pipeline_data.longs]
    shorts = pipeline_data.index[pipeline_data.shorts]

    for asset in longs:
        order_target_percent(asset, 1.0 / 3.0)
    for asset in shorts:
        order_target_percent(asset, -1.0 / 3.0)

    for asset in context.portfolio.positions:
        if asset not in longs and asset not in shorts and data.can_trade(asset):
            order_target_percent(asset, 0)
```

### Example 2: Multi-Factor Ranking

Combine multiple factors with ranking and normalization:

```python
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import AverageDollarVolume, Returns, RSI


def make_pipeline():
    # Factor definitions
    momentum = Returns(window_length=20).rank()
    mean_reversion = -Returns(window_length=5).rank()
    rsi_signal = RSI().rank()

    # Composite score (equal-weighted)
    composite = (momentum + mean_reversion + rsi_signal).rank()

    # Liquidity filter
    liquid = AverageDollarVolume(window_length=30).top(100)

    return Pipeline(
        columns={
            "score": composite,
            "longs": composite.top(10, mask=liquid),
            "shorts": composite.bottom(10, mask=liquid),
        },
        screen=liquid,
    )
```

### Data Ingestion

Zipline supports CSV-based data bundles for any market:

```python
# In ~/.zipline/extension.py
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

register(
    "my-data",
    csvdir_equities(["daily"], "/path/to/csv/dir"),
    calendar_name="XNYS",
)
```

```bash
# Ingest and run
zipline ingest -b my-data
zipline run -f strategy.py --start 2020-1-1 --end 2024-1-1 -o results.pickle --no-benchmark -b my-data
```

More examples in the [examples](src/zipline/examples) directory.

## Tech Stack

| Component | Version | Notes |
|---|---|---|
| **Python** | >= 3.10 | Tested on 3.10 &ndash; 3.13 |
| **pandas** | >= 1.3 | Full NumPy 2.0 support with pandas >= 2.2.2 |
| **NumPy** | >= 1.23 | NumPy 2.x compatible |
| **PyArrow** | >= 14.0 | Parquet I/O with zstd compression |
| **SQLAlchemy** | >= 2.0 | Asset metadata & adjustment storage |
| **exchange_calendars** | >= 4.2 | 42 global market calendars |
| **Cython** | >= 0.29 | Performance-critical components |

## Contributing

This project is sponsored by [Kavout](https://kavout.com). Built upon the work of [Quantopian/zipline](https://github.com/quantopian/zipline) and [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded).

Found a bug or have a suggestion? [Open an issue](https://github.com/teleclaws/zipline-refresh/issues/new).

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
