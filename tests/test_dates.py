import numpy as np
import pytest

from core.dates import parse_dates, infer_dayfirst, DateParseError


def test_variable_width_dayfirst():
    got = parse_dates(["1/2/2020", "2/2/2020", "9/2/2020", "10/2/2020", "12/2/2020"])
    assert list(got.day) == [1, 2, 9, 10, 12]
    assert set(got.month) == {2}
    assert got.is_monotonic_increasing


def test_explicit_format_one_or_two_digits():
    got = parse_dates(["1/2/2020", "12/2/2020"], date_format="%d/%m/%Y")
    assert list(got.day) == [1, 12]
    assert list(got.month) == [2, 2]


def test_iso_dash_month_name_and_time_component():
    got = parse_dates(["2020-01-14", "15-Jan-2020 00:00", "16 January 2020"])
    assert list(got.day) == [14, 15, 16]
    assert set(got.month) == {1}


def test_two_digit_year_via_format():
    got = parse_dates(["01/03/21", "02/03/21"], date_format="%d/%m/%y")
    assert list(got.year) == [2021, 2021]


def test_bad_rows_raise_with_examples():
    with pytest.raises(DateParseError) as err:
        parse_dates(["1/1/2020", "not a date", "3/1/2020"])
    assert "not a date" in str(err.value)


def test_bad_rows_coerced_when_asked():
    got = parse_dates(["1/1/2020", "", "3/1/2020"], coerce=True)
    assert np.asarray(got.isna()).tolist() == [False, True, False]


def test_us_locale_is_flagged_not_scrambled():
    us = [f"03/{d:02d}/2020" for d in range(1, 15)]
    with pytest.raises(DateParseError) as err:
        parse_dates(us)
    assert "chronological" in str(err.value)
    assert "month-first" in str(err.value)


def test_us_locale_parses_when_told():
    us = [f"03/{d:02d}/2020" for d in range(1, 15)]
    got = parse_dates(us, dayfirst=False)
    assert set(got.month) == {3}
    assert got.is_monotonic_increasing


def test_infer_dayfirst():
    assert infer_dayfirst(["13/01/2020", "05/01/2020"]) is True
    assert infer_dayfirst(["01/13/2020", "01/05/2020"]) is False
    assert infer_dayfirst(["01/02/2020", "03/04/2020"]) is None
    assert infer_dayfirst(["2020-01-15", "2020-01-16"]) is None
