import plotly.graph_objects as go

from dataclasses import dataclass


@dataclass
class Line:
    column: str
    color: str
    name: str
    opacity: float = 0.8
    width: int = 1


def plot(df, lines, title="", y_title=""):
    fig = go.Figure()

    for line in lines:
        fig.add_trace(
            go.Scatter(
                x=df["submission_date"],
                y=df[line.column],
                mode="lines",
                line=dict(color=line.color, width=line.width),
                opacity=line.opacity,
                name=line.name,
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        xaxis_title="Submission Date",
        template="plotly_white",
    )

    fig.show()
