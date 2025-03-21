import numpy as np
import pandas as pd

from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from typing import List


@dataclass
class ProductGroup:
    app_names: List[str]


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
