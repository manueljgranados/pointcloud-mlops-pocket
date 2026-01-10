from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def pointcloud_figure(points: np.ndarray, title: str = "Point cloud") -> go.Figure:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=2, opacity=0.8),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def figure_to_json(fig: go.Figure) -> str:
    return fig.to_json()


def figure_to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=True, include_plotlyjs="cdn")
