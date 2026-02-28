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
Parquet-based minute bar storage for OHLCV data.

Replaces the legacy bcolz format with Apache Parquet/Arrow. Uses a wide
format (minutes as rows, sids as columns) with one Parquet file per OHLCV
field.  Stores only actual trading minutes — no fixed-stride padding for
early closes — which eliminates the Cython position-math and exclusion
logic required by bcolz.

On-disk layout::

    minute_equities.parquet/
        _metadata.json      # calendar, sessions, version, minutes_per_day
        open.parquet        # rows = actual trading minutes, cols = sids
        high.parquet
        low.parquet
        close.parquet
        volume.parquet
"""
import json
import logging
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from zipline.data.bar_reader import (
    NoDataForSid,
    NoDataOnDate,
)
from zipline.data.minute_bar_reader import MinuteBarReader
from zipline.utils.calendar_utils import get_calendar
from zipline.utils.cli import maybe_show_progress
from zipline.utils.memoize import lazyval

logger = logging.getLogger("ParquetMinuteBars")

VERSION = 1

FIELDS = ("open", "high", "low", "close", "volume")
OHLC = frozenset(["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class ParquetMinuteBarWriter:
    """Writer for minute OHLCV data in Parquet wide format.

    Parameters
    ----------
    rootdir : str
        Directory in which Parquet files will be written.
    calendar : exchange_calendars.ExchangeCalendar
        Trading calendar.
    start_session : pd.Timestamp
        First trading session (inclusive).
    end_session : pd.Timestamp
        Last trading session (inclusive).
    minutes_per_day : int
        Number of trading minutes per regular session (e.g. 390).
    """

    def __init__(
        self,
        rootdir,
        calendar,
        start_session,
        end_session,
        minutes_per_day,
    ):
        self._rootdir = rootdir
        self._calendar = calendar
        self._start_session = start_session
        self._end_session = end_session
        self._minutes_per_day = minutes_per_day

        # Compute the actual trading minutes for the entire range.
        first_minute = calendar.session_first_minute(start_session)
        last_minute = calendar.session_close(end_session)
        self._minutes = calendar.minutes_in_range(first_minute, last_minute)

    # Progress bar helpers (match BcolzMinuteBarWriter interface)
    progress_bar_message = "Merging minute equity files:"
    progress_bar_item_show_func = staticmethod(lambda x: x if x is None else str(x[0]))

    def write(self, data, show_progress=False, invalid_data_behavior="warn"):
        """Write OHLCV data from per-asset DataFrames.

        Parameters
        ----------
        data : iterable[tuple[int, pd.DataFrame]]
            Pairs of (sid, DataFrame) where each DataFrame has OHLCV
            columns and a DatetimeIndex of trading minutes.
        show_progress : bool
            Whether to display a progress bar.
        invalid_data_behavior : {'warn', 'raise', 'ignore'}
            How to handle data outside representable range.
        """
        os.makedirs(self._rootdir, exist_ok=True)

        # Collect per-sid frames.
        sid_frames = {}
        ctx = maybe_show_progress(
            data,
            show_progress=show_progress,
            item_show_func=self.progress_bar_item_show_func,
            label=self.progress_bar_message,
        )
        with ctx as it:
            for sid, df in it:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    sid_frames[sid] = df

        if not sid_frames:
            self._write_empty()
            return

        # Build wide-format frames: {field: DataFrame(minutes × sids)}.
        all_sids = sorted(sid_frames.keys())
        minutes = self._minutes

        wide_frames = {field: pd.DataFrame(index=minutes) for field in FIELDS}

        for sid in all_sids:
            df = sid_frames[sid]
            # Normalize index to UTC DatetimeIndex.
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            elif str(idx.tz) != "UTC":
                idx = idx.tz_convert("UTC")
            df = df.set_index(idx)

            for field in FIELDS:
                if field in df.columns:
                    wide_frames[field][sid] = df[field].reindex(minutes)

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

        self._write_metadata()

    def write_sid(self, sid, df):
        """Write data for a single sid.

        This is a convenience method for compatibility with the bcolz
        writer interface.  It collects into a single-item write.
        """
        self.write([(sid, df)])

    def _write_empty(self):
        """Write empty Parquet files for an empty dataset."""
        empty_df = pd.DataFrame(index=self._minutes)
        for field in FIELDS:
            table = pa.Table.from_pandas(empty_df, preserve_index=True)
            pq.write_table(
                table,
                os.path.join(self._rootdir, f"{field}.parquet"),
                compression="zstd",
            )
        self._write_metadata()

    def _write_metadata(self):
        """Write JSON metadata sidecar."""
        metadata = {
            "version": VERSION,
            "calendar_name": self._calendar.name,
            "start_session_ns": self._start_session.value,
            "end_session_ns": self._end_session.value,
            "minutes_per_day": self._minutes_per_day,
        }
        path = os.path.join(self._rootdir, "_metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class ParquetMinuteBarReader(MinuteBarReader):
    """Reader for minute OHLCV data stored in Parquet wide format.

    Parameters
    ----------
    rootdir : str
        Path to the directory containing the Parquet files.
    """

    def __init__(self, rootdir):
        self._rootdir = rootdir
        self._field_cache = {}

    # -- Metadata / properties -------------------------------------------

    @lazyval
    def _metadata(self):
        path = os.path.join(self._rootdir, "_metadata.json")
        with open(path) as f:
            return json.load(f)

    @lazyval
    def trading_calendar(self):
        return get_calendar(self._metadata["calendar_name"])

    @property
    def calendar(self):
        return self.trading_calendar

    @lazyval
    def first_trading_day(self):
        return pd.Timestamp(self._metadata["start_session_ns"])

    @lazyval
    def last_available_dt(self):
        end_session = pd.Timestamp(self._metadata["end_session_ns"])
        return self.trading_calendar.session_close(end_session)

    @lazyval
    def sessions(self):
        cal = self.trading_calendar
        start = pd.Timestamp(self._metadata["start_session_ns"])
        end = pd.Timestamp(self._metadata["end_session_ns"])
        return cal.sessions_in_range(start, end)

    @lazyval
    def _minutes(self):
        """DatetimeIndex of all actual trading minutes in the dataset."""
        first_minute = self.trading_calendar.session_first_minute(
            self.first_trading_day
        )
        end_session = pd.Timestamp(self._metadata["end_session_ns"])
        last_minute = self.trading_calendar.session_close(end_session)
        return self.trading_calendar.minutes_in_range(
            first_minute, last_minute,
        )

    # -- SID / field data ------------------------------------------------

    @lazyval
    def _sids(self):
        """Array of int64 sids available in this dataset."""
        table = pq.read_table(
            os.path.join(self._rootdir, "close.parquet"),
        )
        df = table.to_pandas()
        sid_strs = [c for c in df.columns]
        return np.array(sorted(int(s) for s in sid_strs), dtype=np.int64)

    @lazyval
    def _sid_to_col_idx(self):
        """Map from sid (int) to column index in the data arrays."""
        return {sid: i for i, sid in enumerate(self._sids)}

    def _get_field_array(self, field):
        """Load a single OHLCV field on demand.

        Arrays are cached after first load so subsequent accesses are free.
        """
        if field not in self._field_cache:
            sid_cols = [str(s) for s in self._sids]
            path = os.path.join(self._rootdir, f"{field}.parquet")
            table = pq.read_table(path, columns=sid_cols)
            df = table.to_pandas().reindex(columns=sid_cols)
            arr = df.values
            if arr.dtype != np.float64:
                arr = arr.astype(np.float64)
            if field == "volume":
                np.nan_to_num(arr, copy=False, nan=0.0)
            self._field_cache[field] = arr
        return self._field_cache[field]

    # -- Core interface methods ------------------------------------------

    def load_raw_arrays(self, fields, start_dt, end_dt, sids):
        """Load raw OHLCV arrays for the given minute range and assets.

        Parameters
        ----------
        fields : list[str]
            OHLCV field names.
        start_dt : pd.Timestamp
            First minute (inclusive).
        end_dt : pd.Timestamp
            Last minute (inclusive).
        sids : array-like[int]
            Asset sids to load.

        Returns
        -------
        list[np.ndarray]
            One array per field, shape (num_minutes, num_assets), float64.
            OHLC has NaN for missing data. Volume has 0 for missing data.
        """
        minutes = self._minutes

        start_idx = minutes.searchsorted(start_dt)
        end_idx = minutes.searchsorted(end_dt, side="right")
        minute_slice = slice(start_idx, end_idx)

        # Build column indexer for the requested sids.
        sid_to_col = self._sid_to_col_idx
        n_assets = len(sids)
        col_indices = np.empty(n_assets, dtype=np.intp)
        unknown_mask = np.zeros(n_assets, dtype=bool)

        for i, sid in enumerate(sids):
            sid_int = int(sid)
            if sid_int in sid_to_col:
                col_indices[i] = sid_to_col[sid_int]
            else:
                col_indices[i] = 0  # placeholder
                unknown_mask[i] = True

        if unknown_mask.all():
            raise ValueError("At least one valid asset id is required.")

        results = []
        for field in fields:
            data = self._get_field_array(field)
            sliced = data[minute_slice][:, col_indices].copy()

            if field in OHLC:
                sliced[sliced == 0] = np.nan
                if unknown_mask.any():
                    sliced[:, unknown_mask] = np.nan
            else:
                # volume
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
            Trading minute.
        field : str
            OHLCV field name.

        Returns
        -------
        float or int
        """
        sid_int = int(sid)
        if sid_int not in self._sid_to_col_idx:
            raise NoDataForSid(
                "No minute data for sid={0}".format(sid)
            )

        try:
            idx = self._minutes.get_loc(dt)
        except KeyError:
            raise NoDataOnDate(
                "No data for dt={0}".format(dt)
            )

        col_idx = self._sid_to_col_idx[sid_int]
        value = self._get_field_array(field)[idx, col_idx]

        if field != "volume":
            if value == 0 or np.isnan(value):
                return np.nan
            return float(value)
        else:
            return int(value)

    def get_last_traded_dt(self, asset, dt):
        """Find the last minute at or before ``dt`` when ``asset`` traded.

        Parameters
        ----------
        asset : Asset or int
            The asset.
        dt : pd.Timestamp
            The minute to start searching from.

        Returns
        -------
        pd.Timestamp or pd.NaT
        """
        sid_int = int(asset)
        col_idx = self._sid_to_col_idx.get(sid_int)
        if col_idx is None:
            return pd.NaT

        minutes = self._minutes
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        idx = minutes.searchsorted(dt, side="right") - 1
        if idx < 0:
            return pd.NaT

        # Use NumPy vectorized search — C-speed even for large prefixes.
        vol_col = self._get_field_array("volume")[:idx + 1, col_idx]
        nonzero = np.flatnonzero(vol_col)
        if len(nonzero) == 0:
            return pd.NaT
        return minutes[nonzero[-1]]
