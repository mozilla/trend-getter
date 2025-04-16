import numpy as np
import pandas as pd
import sqlglot
import prophet

from dataclasses import dataclass
from functools import reduce
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from trend_getter import holidays
from typing import List, Dict


@dataclass
class Scenario:
    name: str
    slug: str
    reference: str
    start_date: str
    end_date: str
    population: str
    cdf_x: List[float]
    cdf_y: List[float]

    def __post_init__(self) -> None:
        if "_" in self.slug:
            raise ValueError("slug cannot contain '_'.")
        if self.slug in ["total", "other"]:
            raise ValueError("slug cannot use the name 'total' or 'other'.")

        self.start_date = pd.to_datetime(self.start_date)
        self.end_date = pd.to_datetime(self.end_date)
        self.query_select = f"IFNULL({self.population}, FALSE) AS {self.slug},"
        self.cdf = interp1d(
            self.cdf_x, self.cdf_y, kind="linear", bounds_error=False, fill_value=(0, 1)
        )
        self.quantile = interp1d(
            self.cdf_y, self.cdf_x, kind="linear", bounds_error=False, fill_value=(0, 1)
        )

    def sample(self, samples=1):
        return self.quantile(np.random.uniform(0, 1, size=samples))

    def plot_distribution(self):
        fig = px.line(
            x=self.cdf_x + [1],
            y=self.cdf_y + [1],
            template="plotly_white",
            title=f'{self.name}<br><sup>"Y% of scenarios have X% or lower subpopulation DAU decrease"</sup>',
        )
        fig.add_trace(
            go.Scatter(
                x=self.cdf_x + [1], y=self.cdf_y + [1], mode="markers", showlegend=False
            )
        )
        fig.update_traces(
            line=dict(color="black", width=3), marker=dict(size=10, color="black")
        )
        fig.update_layout(
            autosize=False,
            xaxis=dict(
                tickformat=".0%",
                tickangle=0,
                title="Subpopulation DAU Decrease",
                tickmode="array",
                tickvals=self.cdf_x + [1],
            ),
            yaxis=dict(
                tickformat=".1%",
                title="Cumulative Probability",
                tickmode="array",
                tickvals=self.cdf_y + [1],
            ),
        )
        fig.show()


@dataclass
class ScenarioForecasts:
    product_group: List[str]
    scenarios: List[Scenario]
    countries: List[str]
    historical_start_date: str
    historical_end_date: str
    forecast_end_date: str
    project: str = "mozdata"
    number_of_simulations: int = 1000
    metric: str = "dau"

    @property
    def column_names_map(self) -> Dict[str, str]:
        return {"submission_date": "ds", "value": "y"}

    def __post_init__(self) -> None:
        self.scenarios = {i.slug: i for i in self.scenarios}
        self.dates_to_predict = pd.DataFrame(
            {
                "submission_date": pd.date_range(
                    self.historical_end_date, self.forecast_end_date
                ).date[1:]
            }
        )

        self.historical_dfs = {}
        self.historical_forecasts = {}
        self.historical_forecast_models = {}
        self.scaled_historical_forecasts = {}
        self.scenario_forecasts = {}
        self.scenario_percent_impacts = {}

    def _query_(self) -> None:
        if self.metric == "dau":
            query = f"""
            SELECT submission_date,
                IF(country IN ({",".join([f"'{i}'" for i in self.countries])}), country, "ROW") AS country,
                {" ".join([i.query_select for i in self.scenarios.values()])}
                SUM(dau) AS dau,
            FROM `mozdata.telemetry.active_users_aggregates`
            WHERE app_name IN ({",".join([f"'{i}'" for i in self.product_group])})
            AND submission_date BETWEEN "{self.historical_start_date}" AND "{self.historical_end_date}"
            GROUP BY ALL
            ORDER BY {", ".join([str(i + 1) for i in range(len(self.scenarios) + 2)])}
            """
        elif self.metric == "mau":
            query = f"""
            SELECT submission_date,
                IF(country IN ({",".join([f"'{i}'" for i in self.countries])}), country, "ROW") AS country,
                {" ".join([i.query_select for i in self.scenarios.values()])}
                SUM(mau) AS dau,
            FROM `mozdata.telemetry.active_users_aggregates`
            WHERE app_name IN ({",".join([f"'{i}'" for i in self.product_group])})
            AND submission_date BETWEEN "{self.historical_start_date}" AND "{self.historical_end_date}"
            GROUP BY ALL
            ORDER BY {", ".join([str(i + 1) for i in range(len(self.scenarios) + 2)])}
            """

        elif self.metric == "engagement":
            query = f"""
            SELECT submission_date,
                IF(country IN ({",".join([f"'{i}'" for i in self.countries])}), country, "ROW") AS country,
                {" ".join([i.query_select for i in self.scenarios.values()])}
                SUM(dau) / SUM(mau) AS dau,
            FROM `moz-fx-data-shared-prod.telemetry.desktop_engagement`
            WHERE app_name IN ({",".join([f"'{i}'" for i in self.product_group])})
            AND submission_date BETWEEN "{self.historical_start_date}" AND "{self.historical_end_date}"
            AND lifecycle_stage = "existing_users"
            GROUP BY ALL
            ORDER BY {", ".join([str(i + 1) for i in range(len(self.scenarios) + 2)])}
            """

        return sqlglot.transpile(query, read="bigquery", pretty=True)[0]

    def fetch_data(self) -> None:
        query = self._query_()
        print(f"Fetching Data:\n\n{query}")

        df = bigquery.Client(project=self.project).query(query).to_dataframe()

        # ensure submission_date has type 'date'
        # raw["submission_date"] = pd.to_datetime(raw["submission_date"]).dt.date

        cols = list(set(df.columns) - {"submission_date", "country", "dau"})
        df["population"] = (
            df[cols]
            .apply(lambda row: "_".join(col for col in cols if row[col]), axis=1)
            .replace("", "other")  # Replace empty strings with "other"
        )

        # Pivot to wide format
        df = (
            df.pivot_table(
                index=["submission_date", "country"],
                columns="population",
                values="dau",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(columns=None)
            .replace({0: np.nan})
        )

        df["total"] = df.drop(columns=["submission_date", "country"]).sum(axis=1)
        self.historical_dates = df["submission_date"]
        self.population_names = (
            ["total"]
            + sorted(set(df.columns) - {"total", "other", "submission_date", "country"})
            + ["other"]
        )

        sub_populations = []

        for pop in self.population_names:
            a = (
                df.groupby(["submission_date", "country"], as_index=False)[pop]
                .sum(min_count=1)
                .dropna()
            )
            a["dau"] = a[pop]
            hi = holidays.HolidayImpacts(
                df=a,
                forecast_start=self.dates_to_predict["submission_date"].min(),
                forecast_end=self.dates_to_predict["submission_date"].max(),
                # detrend_spike_correction=8.0,
            )
            hi.fit()

            b = hi.all_countries
            sub_populations.append(
                b.rename(columns={"dau": f"{pop}_original", "expected": pop})[
                    ["submission_date", pop, f"{pop}_original"]
                ]
            )

            if pop == "total":
                self.holiday_impacts = hi
                self.future_holiday_impacts = hi.predict()

        self.populations = (
            reduce(
                lambda left, right: pd.merge(
                    left, right, on="submission_date", how="outer"
                ),
                sub_populations,
            )
            .sort_values("submission_date")
            .reset_index(drop=True)
        )

    def _get_historical_forecasts(
        self,
        seed=42,
        changepoint_range=0.8,
        seasonality_prior_scale=0.00825,
        changepoint_prior_scale=0.15983,
    ) -> None:
        sub_populations = []

        print("\nForecasting Populations: ", end="")
        for i in self.population_names:
            print(f"{i}", end=" | ")

            np.random.seed(seed)

            observed_df = self.populations[["submission_date", i]].copy(deep=True)
            observed_df["y"] = observed_df[i]
            self.historical_dfs[i] = observed_df

            params = {
                "daily_seasonality": False,
                "weekly_seasonality": True,
                "yearly_seasonality": len(observed_df.dropna()) > (365 * 2),
                "uncertainty_samples": self.number_of_simulations,
                "changepoint_range": changepoint_range,
                "growth": "logistic",
            }

            if observed_df["y"].max() >= 10e6:
                params["seasonality_prior_scale"] = seasonality_prior_scale
                params["changepoint_prior_scale"] = changepoint_prior_scale

            m = prophet.Prophet(**params)
            self.historical_forecast_models[i] = m

            observed = observed_df.rename(columns=self.column_names_map).copy(deep=True)
            future = self.dates_to_predict.rename(columns=self.column_names_map).copy(
                deep=True
            )

            if "growth" in params:
                if observed_df["y"].max() >= 10e6:
                    cap = observed_df["y"].max() * 2.0
                    floor = observed_df["y"].min() * 0.8
                    observed["cap"] = cap
                    observed["floor"] = floor
                    future["cap"] = cap
                    future["floor"] = floor
                else:
                    cap = observed_df["y"].max() * 2.0
                    observed["cap"] = cap
                    observed["floor"] = 0.0
                    future["cap"] = cap
                    future["floor"] = 0.0

            m.fit(observed)

            forecast_df = pd.DataFrame(m.predictive_samples(future)["yhat"])
            self.historical_forecasts[i] = forecast_df

            if i != "total":
                sub_populations.append(self.historical_forecasts[i])

        self.rescaler = self.historical_forecasts["total"] / sum(sub_populations)

        for i in self.population_names:
            if i == "total":
                self.scaled_historical_forecasts[i] = self.historical_forecasts[i]
            else:
                self.scaled_historical_forecasts[i] = (
                    self.historical_forecasts[i] * self.rescaler
                )
        print("done.")

    def _get_scenario_forecasts(self) -> None:
        start_date = self.populations["submission_date"].min()
        end_date = self.dates_to_predict["submission_date"].max()
        self.all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

        filler = pd.concat(
            [
                self.populations.total * np.nan
                for i in range(self.number_of_simulations)
            ],
            axis=1,
        )
        filler.columns = range(self.number_of_simulations)
        print("Running Scenarios: ", end="")
        for population_name, df in self.scaled_historical_forecasts.items():
            self.scenario_forecasts[population_name] = pd.concat(
                [filler, df.copy(deep=True)]
            ).reset_index(drop=True)

        for scenario_name, s in self.scenarios.items():
            print(f"{scenario_name}", end=" | ")
            samples = s.sample(self.number_of_simulations)
            pct_impacts = pd.DataFrame(
                np.column_stack(
                    [
                        np.interp(
                            self.all_dates,
                            pd.to_datetime(
                                [start_date, s.start_date, s.end_date, end_date]
                            ),
                            [0, 0, i, i],
                        )
                        for i in samples
                    ]
                )
            )
            ix = np.argmax(self.all_dates == self.historical_end_date)
            pct_impacts.iloc[:ix] = 0
            pct_impacts.iloc[ix:] = pct_impacts.iloc[ix:] - pct_impacts.iloc[ix].values
            self.scenario_percent_impacts[scenario_name] = pct_impacts

            for population_name in self.population_names:
                if scenario_name in population_name.split("_"):
                    self.scenario_forecasts[population_name] *= (
                        1 - self.scenario_percent_impacts[scenario_name]
                    )

        self.scenario_forecasts["total"] *= 0
        for population_name in self.population_names:
            if population_name != "total":
                self.scenario_forecasts["total"] += self.scenario_forecasts[
                    population_name
                ]

        print("done.")
