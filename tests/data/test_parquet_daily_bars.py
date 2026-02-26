"""Tests for Parquet daily bar reader/writer.

Reuses the shared _DailyBarsTestCase base class to verify that the
ParquetDailyBarReader passes all the same tests as Bcolz and HDF5 readers.
"""
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from parameterized import parameterized
from zipline.utils.calendar_utils import get_calendar

from zipline.data.bar_reader import (
    NoDataAfterDate,
    NoDataBeforeDate,
    NoDataOnDate,
)
from zipline.data.hdf5_daily_bars import (
    CLOSE,
    VOLUME,
)
from zipline.data.parquet_daily_bars import (
    ParquetDailyBarReader,
    ParquetDailyBarWriter,
)
from zipline.testing.predicates import (
    assert_equal,
    assert_sequence_equal,
)
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithParquetEquityDailyBarReader,
    WithTmpDir,
    WithTradingCalendars,
    ZiplineTestCase,
    WithEquityDailyBarData,
    WithSeededRandomState,
)
from zipline.utils.classproperty import classproperty
from zipline.pipeline.loaders.synthetic import (
    OHLCV,
    expected_bar_value_with_holes,
    make_bar_data,
    asset_start,
    asset_end,
    expected_bar_values_2d,
)

from .test_daily_bars import (
    _DailyBarsTestCase,
    EQUITY_INFO,
    HOLES,
    TEST_CALENDAR_START,
    TEST_CALENDAR_STOP,
    TEST_QUERY_START,
    TEST_QUERY_STOP,
)


class ParquetDailyBarTestCase(
    WithParquetEquityDailyBarReader,
    _DailyBarsTestCase,
):
    EQUITY_DAILY_BAR_COUNTRY_CODES = ["US"]

    @classmethod
    def init_class_fixtures(cls):
        super(ParquetDailyBarTestCase, cls).init_class_fixtures()
        cls.daily_bar_reader = cls.parquet_equity_daily_bar_reader

    def test_read_first_trading_day(self):
        assert self.daily_bar_reader.first_trading_day == self.sessions[0]

    def test_sessions(self):
        assert_equal(self.daily_bar_reader.sessions, self.sessions)

    @parameterized.expand(
        [
            (["open"],),
            (["close", "volume"],),
            (["volume", "high", "low"],),
            (["open", "high", "low", "close", "volume"],),
        ]
    )
    def test_read(self, columns):
        self._check_read_results(
            columns,
            self.assets,
            TEST_QUERY_START,
            TEST_QUERY_STOP,
        )

        assets_array = np.array(self.assets)
        for _ in range(5):
            assets = assets_array.copy()
            self.rand.shuffle(assets)
            assets = assets[: np.random.randint(1, len(assets))]
            self._check_read_results(
                columns,
                assets,
                TEST_QUERY_START,
                TEST_QUERY_STOP,
            )

    def test_start_on_asset_start(self):
        columns = ["high", "volume"]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_start(asset),
                end_date=self.sessions[-1],
            )

    def test_start_on_asset_end(self):
        columns = ["close", "volume"]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_end(asset),
                end_date=self.sessions[-1],
            )

    def test_end_on_asset_start(self):
        columns = ["close", "volume"]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.sessions[0],
                end_date=self.asset_start(asset),
            )

    def test_end_on_asset_end(self):
        columns = [CLOSE, VOLUME]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.sessions[0],
                end_date=self.asset_end(asset),
            )

    def test_read_known_and_unknown_sids(self):
        query_assets = (
            [self.assets[-1] + 1]
            + list(range(self.assets[0], self.assets[-1] + 1))
            + [self.assets[-1] + 3]
        )

        columns = [CLOSE, VOLUME]
        self._check_read_results(
            columns,
            query_assets,
            start_date=TEST_QUERY_START,
            end_date=TEST_QUERY_STOP,
        )

    @parameterized.expand(
        [
            ([],),
            ([2],),
            ([2, 4, 800],),
        ]
    )
    def test_read_only_unknown_sids(self, query_assets):
        columns = [CLOSE, VOLUME]
        with pytest.raises(ValueError):
            self.daily_bar_reader.load_raw_arrays(
                columns,
                TEST_QUERY_START,
                TEST_QUERY_STOP,
                query_assets,
            )

    def test_unadjusted_get_value(self):
        reader = self.daily_bar_reader

        def make_failure_msg(asset, date, field):
            return "Unexpected value for sid={}; date={}; field={}.".format(
                asset, date.date(), field
            )

        for asset in self.assets:
            asset_start = self.asset_start(asset)
            asset_dates = self.dates_for_asset(asset)
            asset_middle = asset_dates[len(asset_dates) // 2]
            asset_end = self.asset_end(asset)

            assert_equal(
                reader.get_value(asset, asset_start, CLOSE),
                expected_bar_value_with_holes(
                    asset_id=asset,
                    date=asset_start,
                    colname=CLOSE,
                    holes=self.holes,
                    missing_value=np.nan,
                ),
                msg=make_failure_msg(asset, asset_start, CLOSE),
            )

            assert_equal(
                reader.get_value(asset, asset_middle, CLOSE),
                expected_bar_value_with_holes(
                    asset_id=asset,
                    date=asset_middle,
                    colname=CLOSE,
                    holes=self.holes,
                    missing_value=np.nan,
                ),
                msg=make_failure_msg(asset, asset_middle, CLOSE),
            )

            assert_equal(
                reader.get_value(asset, asset_end, CLOSE),
                expected_bar_value_with_holes(
                    asset_id=asset,
                    date=asset_end,
                    colname=CLOSE,
                    holes=self.holes,
                    missing_value=np.nan,
                ),
                msg=make_failure_msg(asset, asset_end, CLOSE),
            )

            assert_equal(
                reader.get_value(asset, asset_start, VOLUME),
                expected_bar_value_with_holes(
                    asset_id=asset,
                    date=asset_start,
                    colname=VOLUME,
                    holes=self.holes,
                    missing_value=0,
                ),
                msg=make_failure_msg(asset, asset_start, VOLUME),
            )

    def test_unadjusted_get_value_no_data(self):
        reader = self.daily_bar_reader

        for asset in self.assets:
            before_start = self.trading_calendar.previous_session(
                self.asset_start(asset)
            )
            after_end = self.trading_calendar.next_session(self.asset_end(asset))

            if TEST_CALENDAR_START <= before_start <= TEST_CALENDAR_STOP:
                with pytest.raises(NoDataBeforeDate):
                    reader.get_value(asset, before_start, CLOSE)

            if TEST_CALENDAR_START <= after_end <= TEST_CALENDAR_STOP:
                with pytest.raises(NoDataAfterDate):
                    reader.get_value(asset, after_end, CLOSE)

        for asset, dates in self.holes.items():
            for date in dates:
                assert_equal(
                    reader.get_value(asset, date, CLOSE),
                    np.nan,
                    msg=(
                        "Expected a hole for sid={}; date={}, but got a"
                        " non-nan value for close."
                    ).format(asset, date.date()),
                )
                assert_equal(
                    reader.get_value(asset, date, VOLUME),
                    0.0,
                    msg=(
                        "Expected a hole for sid={}; date={}, but got a"
                        " non-zero value for volume."
                    ).format(asset, date.date()),
                )

    def test_get_last_traded_dt(self):
        for sid in self.assets:
            assert_equal(
                self.daily_bar_reader.get_last_traded_dt(
                    self.asset_finder.retrieve_asset(sid),
                    self.EQUITY_DAILY_BAR_END_DATE,
                ),
                self.asset_end(sid),
            )

            mid_date = pd.Timestamp("2015-06-15")
            if self.asset_start(sid) <= mid_date:
                expected = min(self.asset_end(sid), mid_date)
            else:
                expected = pd.NaT

            assert_equal(
                self.daily_bar_reader.get_last_traded_dt(
                    self.asset_finder.retrieve_asset(sid),
                    mid_date,
                ),
                expected,
            )

            assert_equal(
                self.daily_bar_reader.get_last_traded_dt(
                    self.asset_finder.retrieve_asset(sid),
                    pd.Timestamp(0),
                ),
                pd.NaT,
            )

    def test_listing_currency(self):
        all_assets = np.array(list(self.assets))
        all_results = self.daily_bar_reader.currency_codes(all_assets)
        all_expected = self.make_equity_daily_bar_currency_codes(
            self.DAILY_BARS_TEST_QUERY_COUNTRY_CODE,
            all_assets,
        ).values
        assert_equal(all_results, all_expected)

        assert all_results.dtype == np.dtype(object)
        for code in all_results:
            assert isinstance(code, str)

    def test_listing_currency_for_nonexistent_asset(self):
        reader = self.daily_bar_reader

        valid_sid = max(self.assets)
        valid_currency = reader.currency_codes(np.array([valid_sid]))[0]
        invalid_sids = [-1, -2]

        mixed = np.array(invalid_sids + [valid_sid])
        result = self.daily_bar_reader.currency_codes(mixed)
        expected = np.array([None] * 2 + [valid_currency])
        assert_equal(result, expected)

    def test_invalid_date(self):
        INVALID_DATES = (
            self.trading_calendar.previous_session(TEST_CALENDAR_START),
            pd.Timestamp("2015-06-07", tz="UTC"),
            self.trading_calendar.next_session(TEST_CALENDAR_STOP),
        )

        for invalid_date in INVALID_DATES:
            with pytest.raises(NoDataOnDate):
                self.daily_bar_reader.load_raw_arrays(
                    OHLCV,
                    invalid_date,
                    TEST_QUERY_STOP,
                    self.assets,
                )

            with pytest.raises(NoDataOnDate):
                self.daily_bar_reader.get_value(
                    self.assets[0],
                    invalid_date,
                    "close",
                )


class ParquetDailyBarWriterTestCase(
    WithAssetFinder, WithTmpDir, WithTradingCalendars, ZiplineTestCase
):
    """Test the ParquetDailyBarWriter directly."""

    @classmethod
    def make_equity_info(cls):
        return EQUITY_INFO.loc[EQUITY_INFO.index == 5].copy()

    def test_write_creates_expected_files(self):
        sessions = self.trading_calendar.sessions_in_range(
            TEST_CALENDAR_START,
            TEST_CALENDAR_STOP,
        )

        path = self.tmpdir.makedir("test_write")
        writer = ParquetDailyBarWriter(
            path,
            self.trading_calendar,
            sessions[0],
            sessions[-1],
        )

        bar_data = make_bar_data(self.make_equity_info(), sessions)
        writer.write(bar_data)

        for field in ("open", "high", "low", "close", "volume"):
            assert os.path.exists(os.path.join(path, f"{field}.parquet"))
        assert os.path.exists(os.path.join(path, "lifetimes.parquet"))
        assert os.path.exists(os.path.join(path, "currency.parquet"))
        assert os.path.exists(os.path.join(path, "_metadata.json"))

    def test_write_empty(self):
        sessions = self.trading_calendar.sessions_in_range(
            TEST_CALENDAR_START,
            TEST_CALENDAR_STOP,
        )

        path = self.tmpdir.makedir("test_empty")
        writer = ParquetDailyBarWriter(
            path,
            self.trading_calendar,
            sessions[0],
            sessions[-1],
        )

        writer.write(iter(()))

        reader = ParquetDailyBarReader(path)
        assert_equal(reader.sessions, sessions)

    def test_roundtrip_with_currency_codes(self):
        sessions = self.trading_calendar.sessions_in_range(
            TEST_CALENDAR_START,
            TEST_CALENDAR_STOP,
        )

        path = self.tmpdir.makedir("test_currencies")
        writer = ParquetDailyBarWriter(
            path,
            self.trading_calendar,
            sessions[0],
            sessions[-1],
        )

        bar_data = list(make_bar_data(self.make_equity_info(), sessions))
        currencies = pd.Series(index=[5], data=["CAD"])

        writer.write(iter(bar_data), currency_codes=currencies)

        reader = ParquetDailyBarReader(path)
        result = reader.currency_codes(np.array([5]))
        assert_equal(result, np.array(["CAD"], dtype=object))
