import holidays
import pandas as pd

from datetime import datetime
from dateutil.easter import easter


class PaschalCycleHolidays(holidays.HolidayBase):
    """Custom holiday calendar for the Paschal Cycle holidays."""

    def _populate(self, year):
        # Get Easter Sunday for the given year
        easter_sunday = easter(year)

        # Define Paschal Cycle holidays
        self[easter_sunday - pd.Timedelta(days=47)] = "Mardi Gras"
        self[easter_sunday - pd.Timedelta(days=7)] = "Palm Sunday"
        self[easter_sunday - pd.Timedelta(days=2)] = "Good Friday"
        self[easter_sunday] = "Easter Sunday"
        self[easter_sunday + pd.Timedelta(days=39)] = "Ascension Day"
        self[easter_sunday + pd.Timedelta(days=60)] = "Corpus Christi"

        self[pd.Timestamp(year=year, month=11, day=1)] = "All Saints' Day"


class MozillaHolidays(holidays.HolidayBase):
    """Custom holiday calendar for Mozilla events."""

    def _populate(self, year):
        for i in range(6, 14):
            self[datetime(2019, 5, i).date()] = "Data Loss"

        for i in range(15, 18):
            self[datetime(2019, 7, i).date()] = "Data Loss"
