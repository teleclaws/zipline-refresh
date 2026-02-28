# Zipline Reloaded - Project Context for Claude

## Overview
Zipline is a Pythonic algorithmic trading library for backtesting trading strategies. This is the "reloaded" fork maintained by Stefan Jansen after Quantopian closed in 2020.

## Key Information
- **Python Version**: >= 3.10
- **Main Dependencies**: pandas >= 1.3, SQLAlchemy >= 2, numpy >= 1.23, pyarrow >= 14.0
- **Documentation**: https://zipline.ml4trading.io
- **Community**: https://exchange.ml4trading.io

## Project Structure
- `src/zipline/`: Main source code
  - `algorithm.py`: Core algorithm execution
  - `api.py`: Public API functions
  - `data/`: Data ingestion and handling
    - `parquet_daily_bars.py`: Daily bar storage (Parquet format)
    - `parquet_minute_bars.py`: Minute bar storage (Parquet format)
    - `bar_reader.py`: BarReader ABC, shared constants (`US_EQUITIES_MINUTES_PER_DAY`, `FUTURES_MINUTES_PER_DAY`)
    - `hdf5_daily_bars.py`: HDF5 daily bar storage (multi-country)
    - `bundles/core.py`: Data bundle ingest/load system (Parquet-only)
  - `finance/`: Financial calculations and order execution
  - `pipeline/`: Factor-based screening system
- `tests/`: Test suite
- `docs/`: Documentation source

## Development Commands
```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/test_algorithm.py

# Build documentation
cd docs && make html

# Install in development mode
pip install -e .
```

## Testing Approach
- Unit tests use pytest
- Test data is stored in `tests/resources/`
- Mock trading environments for testing strategies

## Common Tasks
1. **Implementing new data bundles**: See `src/zipline/data/bundles/`
2. **Adding new pipeline factors**: See `src/zipline/pipeline/factors/`
3. **Modifying order execution**: See `src/zipline/finance/execution.py`
4. **Working with trading calendars**: Uses `exchange_calendars` library

## Current Branch
`main` — forked from zipline-reloaded with custom modifications.

## Data Storage Architecture
- **bcolz has been fully removed** (Phase 0-3 migration completed). All bar data uses Apache Parquet via pyarrow.
- Daily bars: `ParquetDailyBarWriter`/`ParquetDailyBarReader` — single `.parquet` file per bundle, wide format (one row per sid×day)
- Minute bars: `ParquetMinuteBarWriter`/`ParquetMinuteBarReader` — directory of per-session `.parquet` files
- Bundle system (`bundles/core.py`) uses Parquet-only paths; no bcolz fallback
- OHLCV data stored as float64 (not uint32 like the old bcolz format)
- Test fixtures: `WithParquetEquityDailyBarReader`, `WithParquetEquityMinuteBarReader`, `WithParquetFutureDailyBarReader`, `WithParquetFutureMinuteBarReader`
- Backward-compat aliases exist: `tmp_bcolz_equity_minute_bar_reader` → `tmp_parquet_equity_minute_bar_reader`

## Important Notes
- The project uses Cython for performance-critical components (`_resample.pyx`, `_adjustments.pyx`, etc.)
- Be careful with numpy/pandas API changes due to major version updates
- Trading calendars are handled by the external `exchange_calendars` package
- Bundle ingest snaps start/end sessions to valid trading days (e.g. 2014-01-01 → 2014-01-02 for NYSE)
- `test_arg_types` in `test_algorithm.py` is skipped on Cython >= 3.0 to avoid segfaults
