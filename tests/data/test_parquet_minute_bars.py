"""Tests for Parquet minute bar reader/writer.

Mirrors the key tests from test_minute_bars.py (bcolz) to verify that the
ParquetMinuteBarReader/Writer passes the same correctness checks.
"""
from datetime import timedelta
import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from zipline.data.bar_reader import NoDataForSid, NoDataOnDate
from zipline.data.bcolz_minute_bars import US_EQUITIES_MINUTES_PER_DAY
from zipline.data.parquet_minute_bars import (
    ParquetMinuteBarReader,
    ParquetMinuteBarWriter,
)
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithInstanceTmpDir,
    WithTradingCalendars,
    ZiplineTestCase,
)

# Calendar range covers half-day sessions (Thanksgiving/Christmas Eve).
TEST_CALENDAR_START = pd.Timestamp("2014-06-02")
TEST_CALENDAR_STOP = pd.Timestamp("2015-12-31")


class ParquetMinuteBarTestCase(
    WithTradingCalendars, WithAssetFinder, WithInstanceTmpDir, ZiplineTestCase
):
    ASSET_FINDER_EQUITY_SIDS = 1, 2

    @classmethod
    def init_class_fixtures(cls):
        super(ParquetMinuteBarTestCase, cls).init_class_fixtures()

        cal = cls.trading_calendar.schedule.loc[
            TEST_CALENDAR_START:TEST_CALENDAR_STOP
        ]
        cls.market_opens = cls.trading_calendar.first_minutes[
            TEST_CALENDAR_START:TEST_CALENDAR_STOP
        ]
        cls.market_closes = cal.close

        cls.test_calendar_start = cls.market_opens.index[0]
        cls.test_calendar_stop = cls.market_opens.index[-1]

    def _make_writer_and_reader(self):
        """Create a fresh writer/reader pair in a temp directory."""
        dest = self.instance_tmpdir.getpath("minute_bars_parquet")
        os.makedirs(dest, exist_ok=True)
        writer = ParquetMinuteBarWriter(
            dest,
            self.trading_calendar,
            TEST_CALENDAR_START,
            TEST_CALENDAR_STOP,
            US_EQUITIES_MINUTES_PER_DAY,
        )
        return dest, writer

    def _write_and_read(self, data_pairs):
        """Write data pairs and return a reader.

        Parameters
        ----------
        data_pairs : list[tuple[int, pd.DataFrame]]
            List of (sid, DataFrame) pairs to write.

        Returns
        -------
        ParquetMinuteBarReader
        """
        dest, writer = self._make_writer_and_reader()
        writer.write(data_pairs)
        return ParquetMinuteBarReader(dest)

    # -- Basic write / read tests ------------------------------------------

    def test_write_one_ohlcv(self):
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        assert 10.0 == reader.get_value(sid, minute, "open")
        assert 20.0 == reader.get_value(sid, minute, "high")
        assert 30.0 == reader.get_value(sid, minute, "low")
        assert 40.0 == reader.get_value(sid, minute, "close")
        assert 50 == reader.get_value(sid, minute, "volume")

    def test_write_two_bars(self):
        minute_0 = self.market_opens[self.test_calendar_start]
        minute_1 = minute_0 + timedelta(minutes=1)
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0, 11.0],
                "high": [20.0, 21.0],
                "low": [30.0, 31.0],
                "close": [40.0, 41.0],
                "volume": [50.0, 51.0],
            },
            index=[minute_0, minute_1],
        )
        reader = self._write_and_read([(sid, data)])

        assert 10.0 == reader.get_value(sid, minute_0, "open")
        assert 20.0 == reader.get_value(sid, minute_0, "high")
        assert 30.0 == reader.get_value(sid, minute_0, "low")
        assert 40.0 == reader.get_value(sid, minute_0, "close")
        assert 50 == reader.get_value(sid, minute_0, "volume")

        assert 11.0 == reader.get_value(sid, minute_1, "open")
        assert 21.0 == reader.get_value(sid, minute_1, "high")
        assert 31.0 == reader.get_value(sid, minute_1, "low")
        assert 41.0 == reader.get_value(sid, minute_1, "close")
        assert 51 == reader.get_value(sid, minute_1, "volume")

    def test_write_on_second_day(self):
        second_day = self.test_calendar_start + timedelta(days=1)
        minute = self.market_opens[second_day]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        assert 10.0 == reader.get_value(sid, minute, "open")
        assert 20.0 == reader.get_value(sid, minute, "high")
        assert 30.0 == reader.get_value(sid, minute, "low")
        assert 40.0 == reader.get_value(sid, minute, "close")
        assert 50 == reader.get_value(sid, minute, "volume")

    def test_write_on_multiple_days(self):
        tds = self.market_opens.index
        days = tds[
            tds.slice_indexer(
                start=self.test_calendar_start + timedelta(days=1),
                end=self.test_calendar_start + timedelta(days=3),
            )
        ]
        minutes = pd.DatetimeIndex(
            [
                self.market_opens[days[0]] + timedelta(minutes=60),
                self.market_opens[days[1]] + timedelta(minutes=120),
            ]
        )
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0, 11.0],
                "high": [20.0, 21.0],
                "low": [30.0, 31.0],
                "close": [40.0, 41.0],
                "volume": [50.0, 51.0],
            },
            index=minutes,
        )
        reader = self._write_and_read([(sid, data)])

        assert 10.0 == reader.get_value(sid, minutes[0], "open")
        assert 20.0 == reader.get_value(sid, minutes[0], "high")
        assert 30.0 == reader.get_value(sid, minutes[0], "low")
        assert 40.0 == reader.get_value(sid, minutes[0], "close")
        assert 50 == reader.get_value(sid, minutes[0], "volume")

        assert 11.0 == reader.get_value(sid, minutes[1], "open")
        assert 21.0 == reader.get_value(sid, minutes[1], "high")
        assert 31.0 == reader.get_value(sid, minutes[1], "low")
        assert 41.0 == reader.get_value(sid, minutes[1], "close")
        assert 51 == reader.get_value(sid, minutes[1], "volume")

    def test_write_empty_dataset(self):
        """Writing an empty iterable should produce a valid reader."""
        dest, writer = self._make_writer_and_reader()
        writer.write([])
        reader = ParquetMinuteBarReader(dest)
        assert reader.first_trading_day == TEST_CALENDAR_START

    def test_write_zeros_read_as_nan(self):
        """OHLC zeros should be read back as NaN; volume zero stays 0."""
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [0.0],
                "high": [0.0],
                "low": [0.0],
                "close": [0.0],
                "volume": [0.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        assert_almost_equal(np.nan, reader.get_value(sid, minute, "open"))
        assert_almost_equal(np.nan, reader.get_value(sid, minute, "high"))
        assert_almost_equal(np.nan, reader.get_value(sid, minute, "low"))
        assert_almost_equal(np.nan, reader.get_value(sid, minute, "close"))
        assert 0 == reader.get_value(sid, minute, "volume")

    # -- Multiple sids -----------------------------------------------------

    def test_write_multiple_sids(self):
        minute = self.market_opens[TEST_CALENDAR_START]
        sids = [1, 2]

        data_1 = pd.DataFrame(
            {
                "open": [15.0],
                "high": [17.0],
                "low": [11.0],
                "close": [15.0],
                "volume": [100.0],
            },
            index=[minute],
        )
        data_2 = pd.DataFrame(
            {
                "open": [25.0],
                "high": [27.0],
                "low": [21.0],
                "close": [25.0],
                "volume": [200.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sids[0], data_1), (sids[1], data_2)])

        assert 15.0 == reader.get_value(sids[0], minute, "open")
        assert 17.0 == reader.get_value(sids[0], minute, "high")
        assert 11.0 == reader.get_value(sids[0], minute, "low")
        assert 15.0 == reader.get_value(sids[0], minute, "close")
        assert 100 == reader.get_value(sids[0], minute, "volume")

        assert 25.0 == reader.get_value(sids[1], minute, "open")
        assert 27.0 == reader.get_value(sids[1], minute, "high")
        assert 21.0 == reader.get_value(sids[1], minute, "low")
        assert 25.0 == reader.get_value(sids[1], minute, "close")
        assert 200 == reader.get_value(sids[1], minute, "volume")

    # -- load_raw_arrays ---------------------------------------------------

    def test_load_raw_arrays(self):
        """Test multi-sid batch read via load_raw_arrays."""
        start_minute = self.market_opens[TEST_CALENDAR_START]
        minutes = [
            start_minute,
            start_minute + pd.Timedelta("1 min"),
            start_minute + pd.Timedelta("2 min"),
        ]
        sids = [1, 2]
        data_1 = pd.DataFrame(
            {
                "open": [15.0, np.nan, 15.1],
                "high": [17.0, np.nan, 17.1],
                "low": [11.0, np.nan, 11.1],
                "close": [14.0, np.nan, 14.1],
                "volume": [1000, 0, 1001],
            },
            index=minutes,
        )
        data_2 = pd.DataFrame(
            {
                "open": [25.0, np.nan, 25.1],
                "high": [27.0, np.nan, 27.1],
                "low": [21.0, np.nan, 21.1],
                "close": [24.0, np.nan, 24.1],
                "volume": [2000, 0, 2001],
            },
            index=minutes,
        )
        reader = self._write_and_read([(sids[0], data_1), (sids[1], data_2)])

        columns = ["open", "high", "low", "close", "volume"]
        arrays = list(
            map(
                np.transpose,
                reader.load_raw_arrays(
                    columns,
                    minutes[0],
                    minutes[-1],
                    sids,
                ),
            )
        )

        data = {sids[0]: data_1, sids[1]: data_2}
        for i, col in enumerate(columns):
            for j, sid in enumerate(sids):
                assert_almost_equal(data[sid][col].values, arrays[i][j])

    def test_load_raw_arrays_early_close(self):
        """Test that early-close sessions don't produce padding rows.

        Since Parquet stores only actual trading minutes, the window that
        spans Thanksgiving (half day) and Christmas Eve (half day) should
        contain exactly the number of actual trading minutes.
        """
        day_before_thanksgiving = pd.Timestamp("2015-11-25")
        xmas_eve = pd.Timestamp("2015-12-24")
        market_day_after_xmas = pd.Timestamp("2015-12-28")

        minutes = [
            self.market_closes[day_before_thanksgiving] - pd.Timedelta("2 min"),
            self.market_closes[xmas_eve] - pd.Timedelta("1 min"),
            self.market_opens[market_day_after_xmas] + pd.Timedelta("1 min"),
        ]
        sids = [1, 2]
        data_1 = pd.DataFrame(
            {
                "open": [15.0, 15.1, 15.2],
                "high": [17.0, 17.1, 17.2],
                "low": [11.0, 11.1, 11.3],
                "close": [14.0, 14.1, 14.2],
                "volume": [1000, 1001, 1002],
            },
            index=minutes,
        )
        data_2 = pd.DataFrame(
            {
                "open": [25.0, 25.1, 25.2],
                "high": [27.0, 27.1, 27.2],
                "low": [21.0, 21.1, 21.2],
                "close": [24.0, 24.1, 24.2],
                "volume": [2000, 2001, 2002],
            },
            index=minutes,
        )
        reader = self._write_and_read([(sids[0], data_1), (sids[1], data_2)])

        columns = ["open", "high", "low", "close", "volume"]
        arrays = reader.load_raw_arrays(
            columns,
            minutes[0],
            minutes[-1],
            sids,
        )

        # Compute expected number of actual trading minutes in the range.
        start_loc = self.trading_calendar.minutes.get_loc(minutes[0])
        end_loc = self.trading_calendar.minutes.get_loc(minutes[-1])
        expected_num_minutes = end_loc - start_loc + 1

        # All returned arrays should have exactly this many rows.
        for arr in arrays:
            assert arr.shape[0] == expected_num_minutes
            assert arr.shape[1] == 2  # two sids

        # Verify actual data values at the three sparse minutes.
        start_minute_loc = self.trading_calendar.minutes.get_loc(minutes[0])
        data = {sids[0]: data_1, sids[1]: data_2}
        for data_minute_idx, minute in enumerate(minutes):
            row_idx = (
                self.trading_calendar.minutes.get_loc(minute) - start_minute_loc
            )
            for col_idx, col in enumerate(columns):
                for sid_idx, sid in enumerate(sids):
                    expected = data[sid][col].iloc[data_minute_idx]
                    actual = arrays[col_idx][row_idx, sid_idx]
                    assert_almost_equal(
                        expected,
                        actual,
                        err_msg=f"Mismatch at {col}/{sid}/{minute}",
                    )

    def test_load_raw_arrays_unknown_sids(self):
        """Unknown sids get NaN for OHLC, 0 for volume."""
        minute = self.market_opens[TEST_CALENDAR_START]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        # Request sid 1 (known) and sid 999 (unknown).
        arrays = reader.load_raw_arrays(
            ["close", "volume"],
            minute,
            minute,
            [sid, 999],
        )
        # Known sid.
        assert 40.0 == arrays[0][0, 0]  # close
        assert 50.0 == arrays[1][0, 0]  # volume

        # Unknown sid.
        assert np.isnan(arrays[0][0, 1])  # close → NaN
        assert 0 == arrays[1][0, 1]  # volume → 0

    def test_load_raw_arrays_all_unknown_raises(self):
        """Requesting only unknown sids should raise ValueError."""
        minute = self.market_opens[TEST_CALENDAR_START]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        with pytest.raises(ValueError):
            reader.load_raw_arrays(
                ["close"],
                minute,
                minute,
                [999, 998],
            )

    # -- NaN handling ------------------------------------------------------

    def test_nans_round_trip(self):
        """NaN OHLC data round-trips correctly; zero volume stays zero."""
        start_minute = self.market_opens[TEST_CALENDAR_START]
        minutes = pd.date_range(start_minute, periods=9, freq="min")
        sid = 1
        data = pd.DataFrame(
            {
                "open": np.full(9, np.nan),
                "high": np.full(9, np.nan),
                "low": np.full(9, np.nan),
                "close": np.full(9, np.nan),
                "volume": np.full(9, 0.0),
            },
            index=minutes,
        )
        reader = self._write_and_read([(sid, data)])

        fields = ["open", "high", "low", "close", "volume"]
        ohlcv_window = list(
            map(
                np.transpose,
                reader.load_raw_arrays(fields, minutes[0], minutes[-1], [sid]),
            )
        )
        for i, field in enumerate(fields):
            if field != "volume":
                assert_array_equal(np.full(9, np.nan), ohlcv_window[i][0])
            else:
                assert_array_equal(np.zeros(9), ohlcv_window[i][0])

    # -- Error cases -------------------------------------------------------

    def test_no_data_for_sid(self):
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        with pytest.raises(NoDataForSid):
            reader.get_value(1337, minute, "close")

    def test_no_data_on_date(self):
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        # Query a minute that is not a trading minute.
        non_trading_minute = pd.Timestamp("2014-06-02 00:00", tz="UTC")
        with pytest.raises(NoDataOnDate):
            reader.get_value(sid, non_trading_minute, "close")

    # -- get_last_traded_dt ------------------------------------------------

    def test_get_last_traded_dt(self):
        start_minute = self.market_opens[TEST_CALENDAR_START]
        minutes = [
            start_minute,
            start_minute + pd.Timedelta("1 min"),
            start_minute + pd.Timedelta("2 min"),
        ]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0, 11.0, np.nan],
                "high": [20.0, 21.0, np.nan],
                "low": [30.0, 31.0, np.nan],
                "close": [40.0, 41.0, np.nan],
                "volume": [100, 200, 0],
            },
            index=minutes,
        )
        reader = self._write_and_read([(sid, data)])

        # At minute_2 (no volume), last traded should be minute_1.
        last = reader.get_last_traded_dt(sid, minutes[2])
        assert last == minutes[1]

        # At minute_1 (has volume), last traded is minute_1 itself.
        last = reader.get_last_traded_dt(sid, minutes[1])
        assert last == minutes[1]

        # At minute_0, last traded is minute_0.
        last = reader.get_last_traded_dt(sid, minutes[0])
        assert last == minutes[0]

    def test_get_last_traded_dt_unknown_sid(self):
        minute = self.market_opens[TEST_CALENDAR_START]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        result = reader.get_last_traded_dt(999, minute)
        assert result is pd.NaT

    # -- Metadata / properties --------------------------------------------

    def test_sessions_and_metadata(self):
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        assert reader.first_trading_day == TEST_CALENDAR_START
        assert reader.trading_calendar.name == self.trading_calendar.name

        expected_sessions = self.trading_calendar.sessions_in_range(
            TEST_CALENDAR_START, TEST_CALENDAR_STOP
        )
        assert_array_equal(reader.sessions, expected_sessions)

        # last_available_dt is the close of the last session.
        expected_last_dt = self.trading_calendar.session_close(TEST_CALENDAR_STOP)
        assert reader.last_available_dt == expected_last_dt

    def test_write_creates_expected_files(self):
        dest, writer = self._make_writer_and_reader()
        minute = self.market_opens[self.test_calendar_start]
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        writer.write([(1, data)])

        for field in ("open", "high", "low", "close", "volume"):
            assert os.path.exists(os.path.join(dest, f"{field}.parquet"))
        assert os.path.exists(os.path.join(dest, "_metadata.json"))

    def test_precision(self):
        """Ensure float64 storage preserves precision (no uint32 scaling)."""
        minute = self.market_opens[self.test_calendar_start]
        sid = 1
        data = pd.DataFrame(
            {
                "open": [130.23],
                "high": [130.23],
                "low": [130.23],
                "close": [130.23],
                "volume": [1000],
            },
            index=[minute],
        )
        reader = self._write_and_read([(sid, data)])

        assert 130.23 == reader.get_value(sid, minute, "open")
        assert 130.23 == reader.get_value(sid, minute, "high")
        assert 130.23 == reader.get_value(sid, minute, "low")
        assert 130.23 == reader.get_value(sid, minute, "close")

    def test_data_frequency(self):
        """Reader should report data_frequency='minute'."""
        minute = self.market_opens[self.test_calendar_start]
        data = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [30.0],
                "close": [40.0],
                "volume": [50.0],
            },
            index=[minute],
        )
        reader = self._write_and_read([(1, data)])
        assert reader.data_frequency == "minute"
