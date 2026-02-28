# Copyright 2026 Zipline Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Parquet-based daily bar storage for OHLCV data.

Replaces the legacy bcolz format with Apache Parquet/Arrow. Uses a wide
format (dates as rows, sids as columns) with one Parquet file per OHLCV
field, following the same logical layout as HDF5DailyBarReader.

On-disk layout::

    daily_equities.parquet/
        _metadata.json      # calendar, sessions, version
        open.parquet        # shape: (num_dates, num_sids), float64
        high.parquet
        low.parquet
        close.parquet
        volume.parquet
        lifetimes.parquet   # columns: sid, start_date, end_date
        currency.parquet    # columns: sid, currency_code
"""
import json
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from zipline.data.bar_reader import (
    NoDataAfterDate,
    NoDataBeforeDate,
    NoDataOnDate,
)
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.utils.calendar_utils import get_calendar
from zipline.utils.cli import maybe_show_progress
from zipline.utils.memoize import lazyval

logger = logging.getLogger("ParquetDailyBars")

VERSION = 1

FIELDS = ("open", "high", "low", "close", "volume")
OHLC = frozenset(["open", "high", "low", "close"])

# Currency code used when no currency information is available.
MISSING_CURRENCY = "XXX"


class ParquetDailyBarWriter:
    """Write daily OHLCV data in Parquet wide format.

    Each OHLCV field is stored as a separate Parquet file with dates as
    rows and sids as columns (float64 for OHLC, float64 for volume).

    Parameters
    ----------
    rootdir : str
        Directory in which to write the Parquet files.
    calendar : TradingCalendar
        Calendar to use for session alignment.
    start_session : pd.Timestamp
        First session (midnight UTC).
    end_session : pd.Timestamp
        Last session (midnight UTC).
    """

    def __init__(self, rootdir, calendar, start_session, end_session):
        self._rootdir = rootdir
        self._calendar = calendar

        start_session = start_session.tz_localize(None)
        end_session = end_session.tz_localize(None)
        self._start_session = start_session
        self._end_session = end_session

        self._sessions = calendar.sessions_in_range(start_session, end_session)

    @property
    def progress_bar_message(self):
        return "Merging daily equity files:"

    def progress_bar_item_show_func(self, value):
        return value if value is None else str(value[0])

    def write(
        self,
        data,
        assets=None,
        show_progress=False,
        invalid_data_behavior="warn",
        currency_codes=None,
    ):
        """Write OHLCV data from per-asset DataFrames.

        Parameters
        ----------
        data : iterable[tuple[int, pd.DataFrame]]
            Pairs of (sid, DataFrame) where each DataFrame has OHLCV
            columns and a DatetimeIndex.
        assets : set[int], optional
            Expected asset ids for progress bar sizing.
        show_progress : bool
            Whether to display a progress bar.
        invalid_data_behavior : {'warn', 'raise', 'ignore'}
            How to handle data outside representable range.
        currency_codes : dict[int, str], optional
            Mapping from sid to currency code. Defaults to USD for all.
        """
        os.makedirs(self._rootdir, exist_ok=True)

        # Collect per-sid frames into wide-format DataFrames.
        sid_frames = {}
        ctx = maybe_show_progress(
            data,
            show_progress=show_progress,
            item_show_func=self.progress_bar_item_show_func,
            label=self.progress_bar_message,
            length=len(assets) if assets is not None else None,
        )
        with ctx as it:
            for sid, df in it:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    sid_frames[sid] = df

        if not sid_frames:
            # Write empty files so downstream readers can open the directory.
            self._write_empty()
            return

        # Build wide-format frames: {field: DataFrame(dates x sids)}
        all_sids = sorted(sid_frames.keys())
        sessions = self._sessions

        wide_frames = {field: pd.DataFrame(index=sessions) for field in FIELDS}

        for sid in all_sids:
            df = sid_frames[sid]
            # Normalize index to tz-naive for alignment with sessions.
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            for field in FIELDS:
                if field in df.columns:
                    wide_frames[field][sid] = df[field].reindex(sessions)

        # Write each field as a Parquet file.
        for field in FIELDS:
            frame = wide_frames[field]
            # Ensure columns are string representations of sids.
            frame.columns = [str(c) for c in frame.columns]
            table = pa.Table.from_pandas(frame, preserve_index=True)
            pq.write_table(
                table,
                os.path.join(self._rootdir, f"{field}.parquet"),
                compression="zstd",
            )

        # Write lifetimes (first/last date with non-NaN data per sid).
        self._write_lifetimes(wide_frames, all_sids, sessions)

        # Write currency codes (default to USD for all sids).
        self._write_currency_codes(all_sids, currency_codes=currency_codes)

        # Write metadata.
        self._write_metadata()

    def write_csvs(
        self,
        asset_map,
        show_progress=False,
        invalid_data_behavior="warn",
    ):
        """Read CSVs and write as Parquet files.

        Parameters
        ----------
        asset_map : dict[int -> str]
            Mapping from asset id to CSV file path.
        show_progress : bool
            Whether to display a progress bar.
        invalid_data_behavior : {'warn', 'raise', 'ignore'}
            How to handle out-of-range data.
        """
        read = partial(
            pd.read_csv,
            parse_dates=["day"],
            index_col="day",
            dtype={
                "open": np.float64,
                "high": np.float64,
                "low": np.float64,
                "close": np.float64,
                "volume": np.float64,
            },
        )
        return self.write(
            ((asset, read(path)) for asset, path in asset_map.items()),
            assets=asset_map.keys(),
            show_progress=show_progress,
            invalid_data_behavior=invalid_data_behavior,
        )

    def _write_empty(self):
        """Write empty Parquet files for an empty dataset."""
        empty_df = pd.DataFrame(index=self._sessions)
        for field in FIELDS:
            table = pa.Table.from_pandas(empty_df, preserve_index=True)
            pq.write_table(
                table,
                os.path.join(self._rootdir, f"{field}.parquet"),
                compression="zstd",
            )

        # Empty lifetimes.
        lifetimes = pd.DataFrame(
            {"sid": pd.array([], dtype="int64"),
             "start_date": pd.array([], dtype="datetime64[ns]"),
             "end_date": pd.array([], dtype="datetime64[ns]")}
        )
        pq.write_table(
            pa.Table.from_pandas(lifetimes, preserve_index=False),
            os.path.join(self._rootdir, "lifetimes.parquet"),
        )

        # Empty currency codes.
        currency = pd.DataFrame(
            {"sid": pd.array([], dtype="int64"),
             "currency_code": pd.array([], dtype="object")}
        )
        pq.write_table(
            pa.Table.from_pandas(currency, preserve_index=False),
            os.path.join(self._rootdir, "currency.parquet"),
        )

        self._write_metadata()

    def _write_lifetimes(self, wide_frames, all_sids, sessions):
        """Compute and write asset lifetimes (first/last date with data)."""
        # Use close field to determine lifetimes.
        close = wide_frames["close"]
        start_dates = []
        end_dates = []
        for sid in all_sids:
            col = str(sid)
            if col in close.columns:
                valid = close[col].dropna()
                if len(valid) > 0:
                    start_dates.append(valid.index[0])
                    end_dates.append(valid.index[-1])
                else:
                    start_dates.append(pd.NaT)
                    end_dates.append(pd.NaT)
            else:
                start_dates.append(pd.NaT)
                end_dates.append(pd.NaT)

        lifetimes = pd.DataFrame({
            "sid": all_sids,
            "start_date": start_dates,
            "end_date": end_dates,
        })
        pq.write_table(
            pa.Table.from_pandas(lifetimes, preserve_index=False),
            os.path.join(self._rootdir, "lifetimes.parquet"),
        )

    def _write_currency_codes(self, all_sids, currency_codes=None):
        """Write currency codes per sid.

        Parameters
        ----------
        all_sids : list[int]
            All sids being written.
        currency_codes : dict-like or pd.Series, optional
            Mapping from sid to currency code string. If None, defaults
            to "USD" for all sids.
        """
        if currency_codes is None:
            codes = ["USD"] * len(all_sids)
        else:
            codes = [currency_codes.get(sid, MISSING_CURRENCY) for sid in all_sids]

        currency = pd.DataFrame({
            "sid": all_sids,
            "currency_code": codes,
        })
        pq.write_table(
            pa.Table.from_pandas(currency, preserve_index=False),
            os.path.join(self._rootdir, "currency.parquet"),
        )

    def _write_metadata(self):
        """Write JSON metadata sidecar."""
        metadata = {
            "version": VERSION,
            "calendar_name": self._calendar.name,
            "start_session_ns": self._start_session.value,
            "end_session_ns": self._end_session.value,
        }
        with open(os.path.join(self._rootdir, "_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


class ParquetDailyBarReader(CurrencyAwareSessionBarReader):
    """Reader for daily OHLCV data stored in Parquet wide format.

    Parameters
    ----------
    rootdir : str
        Path to the directory containing the Parquet files.
    """

    @lazyval
    def _metadata(self):
        path = os.path.join(self._rootdir, "_metadata.json")
        with open(path) as f:
            return json.load(f)

    @lazyval
    def sessions(self):
        cal = get_calendar(self._metadata["calendar_name"])
        start = pd.Timestamp(self._metadata["start_session_ns"])
        end = pd.Timestamp(self._metadata["end_session_ns"])
        return cal.sessions_in_range(start, end)

    @lazyval
    def first_trading_day(self):
        if len(self.sessions) > 0:
            return self.sessions[0]
        return None

    @lazyval
    def trading_calendar(self):
        return get_calendar(self._metadata["calendar_name"])

    @property
    def last_available_dt(self):
        if len(self.sessions) > 0:
            return self.sessions[-1]
        return None

    @lazyval
    def _sids(self):
        """Array of int64 sids available in this dataset."""
        table = pq.read_table(
            os.path.join(self._rootdir, "close.parquet"),
        )
        df = table.to_pandas()
        # Columns are string sids; index is date.
        sid_strs = [c for c in df.columns]
        return np.array(sorted(int(s) for s in sid_strs), dtype=np.int64)

    @lazyval
    def _sid_to_col_idx(self):
        """Map from sid (int) to column index in the data arrays."""
        return {sid: i for i, sid in enumerate(self._sids)}

    def __init__(self, rootdir):
        self._rootdir = rootdir
        self._field_cache = {}

    def _get_field_array(self, field):
        """Load a single OHLCV field on demand.

        Arrays are cached after first load so subsequent accesses are free.
        """
        if field not in self._field_cache:
            sid_cols = [str(s) for s in self._sids]
            path = os.path.join(self._rootdir, f"{field}.parquet")
            table = pq.read_table(path, columns=sid_cols)
            # Reindex to canonical sid order (adds NaN cols for missing sids).
            df = table.to_pandas().reindex(columns=sid_cols)
            arr = df.values
            if arr.dtype != np.float64:
                arr = arr.astype(np.float64)
            if field == "volume":
                np.nan_to_num(arr, copy=False, nan=0.0)
            self._field_cache[field] = arr
        return self._field_cache[field]

    @lazyval
    def _lifetimes(self):
        """DataFrame with columns: sid, start_date, end_date."""
        path = os.path.join(self._rootdir, "lifetimes.parquet")
        return pq.read_table(path).to_pandas()

    @lazyval
    def _lifetimes_map(self):
        """Dict mapping sid (int) to (start_date, end_date) timestamps."""
        df = self._lifetimes
        sids = df["sid"].values.astype(np.int64)
        starts = pd.to_datetime(df["start_date"].values)
        ends = pd.to_datetime(df["end_date"].values)
        return dict(zip(sids, zip(starts, ends)))

    @lazyval
    def _currency_codes_df(self):
        """DataFrame with columns: sid, currency_code."""
        path = os.path.join(self._rootdir, "currency.parquet")
        return pq.read_table(path).to_pandas()

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        """Load raw OHLCV arrays for the given date range and assets.

        Parameters
        ----------
        columns : list[str]
            OHLCV field names.
        start_date : pd.Timestamp
            First date (inclusive).
        end_date : pd.Timestamp
            Last date (inclusive).
        assets : array-like[int]
            Asset sids to load.

        Returns
        -------
        list[np.ndarray]
            One array per column, shape (num_dates, num_assets), float64.
            OHLC has NaN for missing data. Volume has 0 for missing data.
        """
        start_idx = self._date_to_index(start_date)
        end_idx = self._date_to_index(end_date)
        date_slice = slice(start_idx, end_idx + 1)

        # Build a column indexer for the requested assets.
        sid_to_col = self._sid_to_col_idx
        n_assets = len(assets)
        col_indices = np.empty(n_assets, dtype=np.intp)
        unknown_mask = np.zeros(n_assets, dtype=bool)

        for i, sid in enumerate(assets):
            sid_int = int(sid)
            if sid_int in sid_to_col:
                col_indices[i] = sid_to_col[sid_int]
            else:
                col_indices[i] = 0  # placeholder, will be zeroed out
                unknown_mask[i] = True

        if unknown_mask.all():
            raise ValueError("At least one valid asset id is required.")

        n_dates = end_idx - start_idx + 1
        results = []
        for column in columns:
            data = self._get_field_array(column)
            sliced = data[date_slice][:, col_indices].copy()

            if column in OHLC:
                # Zero means no data for OHLC, convert to NaN.
                sliced[sliced == 0] = np.nan
                # Unknown assets get NaN.
                if unknown_mask.any():
                    sliced[:, unknown_mask] = np.nan
            else:
                # Volume: unknown assets get 0.
                if unknown_mask.any():
                    sliced[:, unknown_mask] = 0

            results.append(sliced)

        return results

    def get_value(self, sid, dt, field):
        """Retrieve a single value at the given coordinates.

        Parameters
        ----------
        sid : int
            Asset identifier.
        dt : pd.Timestamp
            Session date.
        field : str
            OHLCV field name.

        Returns
        -------
        float or int
            The price (float for OHLC) or volume (int).

        Raises
        ------
        NoDataOnDate
            If ``dt`` is not a valid session.
        NoDataBeforeDate
            If ``dt`` is before the asset's first trading day.
        NoDataAfterDate
            If ``dt`` is after the asset's last trading day.
        """
        sid_int = int(sid)
        if sid_int not in self._sid_to_col_idx:
            raise NoDataOnDate(
                "No data for sid={0} on dt={1}".format(sid, dt)
            )

        date_idx = self._date_to_index(dt)
        col_idx = self._sid_to_col_idx[sid_int]
        value = self._get_field_array(field)[date_idx, col_idx]

        if field != "volume":
            if value == 0 or np.isnan(value):
                # Check if this is outside the asset's lifetime.
                lifetime = self._lifetimes_map.get(sid_int)
                if lifetime is not None:
                    start, end = lifetime
                    if dt < start:
                        raise NoDataBeforeDate()
                    if dt > end:
                        raise NoDataAfterDate()
                # It's a hole within the asset's lifetime — return NaN.
                return np.nan
            return float(value)
        else:
            if value == 0:
                # Check if this is outside the asset's lifetime.
                lifetime = self._lifetimes_map.get(sid_int)
                if lifetime is not None:
                    start, end = lifetime
                    if dt < start:
                        raise NoDataBeforeDate()
                    if dt > end:
                        raise NoDataAfterDate()
            return int(value)

    def get_last_traded_dt(self, asset, day):
        """Find the last session at or before ``day`` when ``asset`` traded.

        Parameters
        ----------
        asset : Asset
            The asset.
        day : pd.Timestamp
            The session to start searching from.

        Returns
        -------
        pd.Timestamp or pd.NaT
        """
        sid_int = int(asset)
        if sid_int not in self._sid_to_col_idx:
            return pd.NaT

        col_idx = self._sid_to_col_idx[sid_int]
        volumes = self._get_field_array("volume")

        try:
            day_idx = self._date_to_index(day)
        except NoDataOnDate:
            return pd.NaT

        # Walk backward to find a session with volume.
        while day_idx >= 0:
            if volumes[day_idx, col_idx] != 0:
                return self.sessions[day_idx]
            day_idx -= 1

        return pd.NaT

    def currency_codes(self, sids):
        """Get currency codes for the requested sids.

        Parameters
        ----------
        sids : np.array[int64]
            Array of sids.

        Returns
        -------
        np.array[object]
            Currency codes. None for unknown sids.
        """
        df = self._currency_codes_df
        if df.empty:
            return np.array([None] * len(sids), dtype=object)

        code_map = dict(zip(df["sid"].values, df["currency_code"].values))
        return np.array(
            [code_map.get(int(sid)) for sid in sids],
            dtype=object,
        )

    def _date_to_index(self, date):
        """Convert a date to an index in the sessions array.

        Raises NoDataOnDate if date is not in sessions.
        """
        try:
            return self.sessions.get_loc(date)
        except KeyError as exc:
            raise NoDataOnDate(date) from exc
