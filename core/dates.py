# core/dates.py
# Tolerant parsing of a date column to a DatetimeIndex.
#
# The awkward cases in practice are day-first vs month-first ambiguity
# (1/2/2020), variable-width fields (1/2/2020 next to 12/11/2020), mixed
# separators, an ISO row or a trailing time component slipped into an otherwise
# uniform column, two-digit years, and a stray blank or text row. A single
# strptime format with errors='raise' fails the whole file on any of these.
#
# Strategy: parse each value independently (pandas format='mixed') with a
# day-first / month-first preference, or with an explicit format when the caller
# supplies one; then check the result. Unparseable values, and a series that is
# not in chronological order (the usual sign that day and month are swapped),
# raise DateParseError with the offending values and, where it can be worked out,
# the interpretation that does parse in order.

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_BLANKS = {"", "nan", "nat", "none", "null", "na", "n/a", "-"}


class DateParseError(ValueError):
    """A date column could not be parsed unambiguously and in order."""


def _swap_format(fmt):
    """Swap the day and month directives in a strptime format string."""
    return (fmt.replace("%d", "\x00").replace("%-d", "\x00")
               .replace("%m", "%d").replace("%-m", "%d")
               .replace("\x00", "%m"))


def _to_datetime(strings, *, dayfirst, date_format):
    if date_format:
        return pd.to_datetime(strings, format=date_format, errors="coerce")
    # format='mixed' parses each element on its own, so mixed widths, mixed
    # separators, ISO rows and time components are all tolerated.
    return pd.to_datetime(strings, dayfirst=dayfirst, format="mixed",
                          errors="coerce")


def infer_dayfirst(values):
    """Guess day-first (True) or month-first (False) from the values themselves.

    Returns None when the column is genuinely ambiguous (every first and second
    field <= 12), contradictory (some rows force each way), or year-first / ISO.
    """
    day_first = month_first = 0
    for value in pd.Series(values).dropna().astype(str):
        parts = re.findall(r"\d+", value)[:3]
        if len(parts) < 3:
            continue
        a, b, c = (int(parts[0]), int(parts[1]), int(parts[2]))
        if a > 31 or c <= 31:            # looks year-first / ISO; no d-m vote
            continue
        if a > 12 and b <= 12:
            day_first += 1
        elif b > 12 and a <= 12:
            month_first += 1
    if day_first and not month_first:
        return True
    if month_first and not day_first:
        return False
    return None


def parse_dates(values, *, dayfirst=True, date_format=None, coerce=False):
    """Parse `values` (a column of date strings) to a pandas DatetimeIndex.

    Parameters
    ----------
    dayfirst : prefer day-first (Australian / European) when a value is
        ambiguous. Ignored when date_format is given.
    date_format : an explicit strptime pattern such as '%d/%m/%Y'. %d and %m
        accept one or two digits, so '1/2/2020' and '12/2/2020' both parse.
    coerce : if True, values that cannot be parsed become NaT and are returned
        rather than raising (the caller is expected to drop those rows). A
        non-chronological result still raises regardless.

    Raises
    ------
    DateParseError
        when values cannot be read (and coerce is False), or when the parsed
        series runs backwards in time (day and month most likely swapped).
    """
    series = (pd.Series(values, dtype="object").astype("string").str.strip())
    blank = series.isna() | series.str.lower().isin(_BLANKS)
    series = series.mask(blank)

    parsed = _to_datetime(series, dayfirst=dayfirst, date_format=date_format)

    unparsed = parsed.isna() & ~series.isna()
    if unparsed.any() and not coerce:
        examples = list(dict.fromkeys(series[unparsed].dropna().tolist()))[:5]
        how = (f"format {date_format!r}" if date_format
               else f"day-first={dayfirst}")
        raise DateParseError(
            f"{int(unparsed.sum())} date value(s) could not be read with {how}: "
            + ", ".join(repr(x) for x in examples)
        )

    index = pd.DatetimeIndex(parsed)
    good = index[index.notna()]
    if len(good) >= 3 and not good.is_monotonic_increasing:
        backwards = int((good[1:] < good[:-1]).sum())
        hint = ""
        alt_format = _swap_format(date_format) if date_format else None
        alt = _to_datetime(series, dayfirst=not dayfirst, date_format=alt_format)
        alt_good = pd.DatetimeIndex(alt)[pd.DatetimeIndex(alt).notna()]
        if len(alt_good) >= 3 and alt_good.is_monotonic_increasing:
            other = "month-first" if (dayfirst and not date_format) else "day-first"
            hint = (f" They are chronological when read {other}"
                    + (f" (format {alt_format!r})" if alt_format else "")
                    + " - change the date-format setting.")
        raise DateParseError(
            f"The parsed dates are not in chronological order ({backwards} step(s) "
            "go backwards), which usually means the day and month are swapped."
            + hint
        )

    return index
