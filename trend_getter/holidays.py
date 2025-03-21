import holidays
import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import datetime
from dateutil.easter import easter
from typing import Optional


class PaschalCycleHolidays(holidays.HolidayBase):
    """
    Custom holiday calendar for key Christian holidays, mostly related to the
    Paschal cycle. Includes Mardi Gras, Palm Sunday, Good Friday, Easter,
    Ascension Day, Corpus Christi. Also includes All Saints' Day, which is not
    part of the Paschal Cycle.
    """

    def _populate(self, year):
        # Get Easter Sunday for the given year
        easter_sunday = easter(year)

        # Add pre- and post-Easter holidays
        self[easter_sunday - pd.Timedelta(days=47)] = "Mardi Gras"
        self[easter_sunday - pd.Timedelta(days=7)] = "Palm Sunday"
        self[easter_sunday - pd.Timedelta(days=2)] = "Good Friday"
        self[easter_sunday] = "Easter Sunday"
        self[easter_sunday + pd.Timedelta(days=39)] = "Ascension Day"
        self[easter_sunday + pd.Timedelta(days=60)] = "Corpus Christi"

        # Fixed-date holiday
        self[pd.Timestamp(year=year, month=11, day=1)] = "All Saints' Day"


class MozillaHolidays(holidays.HolidayBase):
    """
    Custom holiday calendar for Mozilla-specific historical incidents.
    Currently includes Data Loss events in May and July 2019.
    """

    def _populate(self, year):
        # Only populate holidays if the year includes 2019
        if year == 2019:
            # Data Loss in May 6–13, 2019
            for day in range(6, 14):
                self[datetime(2019, 5, day).date()] = "Data Loss"

            # Additional Data Loss in July 15–17, 2019
            for day in range(15, 18):
                self[datetime(2019, 7, day).date()] = "Data Loss"


def get_calendar(
    country: str,
    holiday_years: list,
    include_paschal_cycle: bool = False,
    split_concurrent_holidays: bool = False,
) -> pd.DataFrame:
    """
    Generate a cleaned and formatted DataFrame of holidays for a specific country.

    Args:
        country (str): Country code (e.g., "US", "FR", "ROW").
        holiday_years (list): List of years to include holidays from.
        include_paschal_cycle (bool): Whether to include holidays like Easter and related observances.
        split_concurrent_holidays (bool): Whether to split semicolon-delimited holidays into multiple rows.

    Returns:
        pd.DataFrame: A DataFrame with holidays, labeled by date, country, and cleaned holiday name.
    """

    # Use US holidays as a default for ROW (Rest of World)
    if country == "ROW":
        country_holidays = holidays.US(years=holiday_years)
    else:
        country_holidays = getattr(holidays, country)(years=holiday_years)

    # Optionally add Paschal cycle holidays
    if include_paschal_cycle:
        country_holidays += PaschalCycleHolidays(years=holiday_years)

    # Include Mozilla-specific holidays for 2019
    if 2019 in holiday_years:
        country_holidays += MozillaHolidays(years=holiday_years)

    # Convert holiday dictionary into DataFrame
    df = pd.DataFrame(
        {
            "submission_date": list(country_holidays.keys()),
            "holiday": list(country_holidays.values()),
            "country": country,
        }
    )

    # Clean holiday name text
    df["holiday"] = df["holiday"].str.replace(
        r"Day off \(substituted.*", "day off (substituted)", regex=True
    )
    df["holiday"] = df["holiday"].str.replace(" (observed)", "", regex=False)

    # Split concurrent holidays into separate rows if requested
    if split_concurrent_holidays:
        df = df.assign(holiday=df["holiday"].str.split("; ")).explode(
            "holiday", ignore_index=True
        )
    else:
        # Otherwise, append the country name to each concurrent holiday
        df["holiday"] = df["holiday"].str.replace(";", f"; {country}", regex=False)

    # Prefix holiday names with country for clarity
    df["holiday"] = df["country"] + " " + df["holiday"]

    df["submission_date"] = pd.to_datetime(df["submission_date"])

    return df.sort_values(by="submission_date").reset_index(drop=True)


def detrend(
    df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    threshold: float = 0.05,
    max_radius: int = 7,
    min_radius: int = 3,
    spike_correction: Optional[float] = None,
) -> pd.DataFrame:
    """
    Applies a physics-inspired detrending algorithm to smooth out holiday-driven dips in DAU.

    Parameters:
        df (pd.DataFrame): DataFrame with submission_date and dau columns.
        holiday_df (pd.DataFrame): DataFrame with submission_date and holiday columns.
        threshold (float): Minimum relative difference for adjustment to apply.
        max_radius (int): Maximum distance (in days) from a holiday for adjustment.
        min_radius (int): Minimum difference (in days) that is meaningful for smoothing.
            Must be 1 <= min_radius <= max_radius. A value of 1 means that the holiday +/-1
            day all have equal weight.
        spike_correction (Optional float): Correction multiple to clamp x, v, or a values.

    Returns:
        pd.DataFrame: Input dataframe with additional columns:
            'x' (position), 'v' (velocity), 'a' (acceleration), and 'expected' (detrended DAU).
    """

    df = df.copy()

    # Calculate a spike-correction factor:
    def scaled_sigmoid(x, L=1.5, U=8, x0=5e5, k=1e-5):
        return L + (U - L) / (1 + np.exp(-k * (x - x0)))

    if spike_correction is None:
        spike_correction = scaled_sigmoid(df.dau.max())

    # Merge in holiday labels
    df["submission_date"] = pd.to_datetime(df["submission_date"])
    df = df.merge(holiday_df, how="left", on="submission_date")

    # Create a holiday date lookup
    holiday_dates = df.loc[df["holiday"].notna(), "submission_date"]

    # Compute days from the nearest holiday
    df["days_from_holiday"] = df["submission_date"].apply(
        lambda date: max((holiday_dates - date).abs().min().days, min_radius)
    )

    # Initialize series for kinematic components and expected values
    _x, _v, _a, _e = [], [], [], []

    for i in df.itertuples():
        idx = i.Index

        if idx >= 21:
            # Get lagged expected values for position/velocity/acceleration
            lag07, lag14, lag21 = _e[idx - 7], _e[idx - 14], _e[idx - 21]

            x = lag07
            v = lag07 - lag14
            a = lag07 - 2 * lag14 + lag21

            # Compute rolling averages for recent values
            x_bar = np.mean(_x[-7:])
            v_bar = np.mean(_v[-7:])
            a_bar = np.mean(_a[-7:])

            # Clamp spikes using relative thresholds
            if abs(x) > spike_correction * abs(x_bar):
                x = x_bar
            if abs(v) > spike_correction * abs(v_bar):
                v = v_bar
            if abs(a) > spike_correction * abs(a_bar):
                a = a_bar

            # Compute expected DAU using position + velocity + 0.5 * acceleration
            e = x + v + 0.5 * a

            _x.append(x)
            _v.append(v)
            _a.append(a)

            # If within holiday radius and relative error is below threshold, apply smoothing
            if (i.days_from_holiday <= max_radius) and (i.dau / abs(e) - 1) < threshold:
                weight = (min_radius + 1) / (i.days_from_holiday + 1)
                blended = e * weight + i.dau * (1 - weight)
                _e.append(blended)
            else:
                _e.append(i.dau)
        else:
            # For early points, fall back to observed DAU
            _x.append(np.nan)
            _v.append(np.nan)
            _a.append(np.nan)
            _e.append(i.dau)

    # Attach kinematic components and expected values to DataFrame
    df["x"] = _x
    df["v"] = _v
    df["a"] = _a
    df["expected"] = _e

    return df


def estimate_impacts(
    dau_dfs: dict,
    holiday_dfs: dict,
    last_observed_date=None,
    dau_column: str = "dau",
    expected_dau_column: str = "expected",
) -> dict:
    """
    Estimate holiday impacts by comparing observed and expected DAU near holidays.

    Parameters:
        dau_dfs (dict): Dictionary of DataFrames with actual and expected DAU per country.
        holiday_dfs (dict): Dictionary of DataFrames with holidays per country.
        last_observed_date (str or pd.Timestamp, optional): Filter out dates >= this value.
        dau_column (str): Column name for actual DAU.
        expected_dau_column (str): Column name for expected DAU.

    Returns:
        dict: Nested dictionary of estimated holiday impacts:
              {holiday: {date_diff: {"impact": [...], "years": set(), "average_impact": float}}}
    """
    holiday_impacts = defaultdict(
        lambda: defaultdict(lambda: {"impact": [], "years": set()})
    )

    print("Calculating holiday impacts for: ", end="")

    for country in dau_dfs:
        print(country, end=", ")

        dau_df = dau_dfs[country].copy()
        holiday_df = holiday_dfs[country].copy()

        # Optional filter to exclude future dates
        if last_observed_date is not None:
            dau_df = dau_df[
                dau_df["submission_date"] < pd.to_datetime(last_observed_date)
            ].copy()

        # Cross-join DAU and holiday dates
        merged_df = dau_df.merge(holiday_df, how="cross", suffixes=("_dau", "_holiday"))

        # Calculate date difference between submission_dates and holiday dates
        merged_df["date_diff"] = (
            merged_df["submission_date_dau"] - merged_df["submission_date_holiday"]
        ).dt.days

        # Keep only rows where a holiday is within ±7 days of the date
        merged_df = merged_df[merged_df["date_diff"].between(-7, 7)].copy()

        # Exclude rows with "Data Loss" holidays
        merged_df = merged_df[
            ~merged_df["holiday_holiday"].str.contains("Data Loss", na=False)
        ].copy()

        # Calculate the DAU impact: (observed - expected)
        merged_df["impact"] = merged_df.groupby("submission_date_dau")[
            dau_column
        ].transform("first") - merged_df.groupby("submission_date_dau")[
            expected_dau_column
        ].transform(
            "first"
        )

        # Apply inverse-distance weighting by date offset
        merged_df["weight"] = 1 / (1 + merged_df["date_diff"].abs())
        merged_df["scale"] = merged_df["weight"] / merged_df.groupby(
            "submission_date_dau"
        )["weight"].transform("sum")

        # Scale the impact by the weight
        merged_df["impact"] *= merged_df["scale"]

        # Expand semicolon-delimited holidays into separate rows
        merged_df = (
            merged_df.assign(holiday=merged_df["holiday_dau"].str.split("; "))
            .explode("holiday")
            .assign(holiday=merged_df["holiday_holiday"].str.split("; "))
            .explode("holiday")
        ).copy()

        # Aggregate impacts into the nested dictionary
        for row in merged_df.itertuples():
            holiday_impacts[row.holiday][row.date_diff]["impact"].append(row.impact)

            # For substituted holidays, store full date; otherwise, store year only
            if "day off (substituted)" in row.holiday:
                holiday_impacts[row.holiday][row.date_diff]["years"].add(
                    row.submission_date_holiday
                )
            else:
                holiday_impacts[row.holiday][row.date_diff]["years"].add(
                    row.submission_date_holiday.year
                )

    # Compute average impact for each (holiday, date_diff) pair
    for diffs in holiday_impacts.values():
        for data in diffs.values():
            if len(data["years"]) > 0:
                data["average_impact"] = sum(data["impact"]) / len(data["years"])
            else:
                data["average_impact"] = 0.0  # Prevent division by zero

    return holiday_impacts
