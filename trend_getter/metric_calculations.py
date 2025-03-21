import pandas as pd


def moving_average(series, window=28):
    return series.rolling(window=window).mean()


def year_over_year(df, column, date_column="submission_date"):
    # Assume df has columns: "submission_date" (datetime), "x"
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Create a copy with submission_date shifted 1 year
    df_shift = df.copy()
    df_shift[date_column] = df_shift[date_column] + pd.DateOffset(years=1)
    df_shift.rename(columns={column: f"{column}_last_year"}, inplace=True)

    # Merge original with shifted version
    df_merged = df.merge(df_shift, on=date_column, how="left")

    # Calculate YoY % change
    return (df_merged[column] / df_merged[f"{column}_last_year"]) - 1
