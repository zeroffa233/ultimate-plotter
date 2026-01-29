"""
Line chart implementation.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List

from .base import BaseChart
from ..style import AcademicStyle


class LineChart(BaseChart):
    """
    Line chart for displaying trends over continuous variables.
    
    Suitable for time series, sequential data, or any continuous
    X-axis variable with one or more Y variables.
    """
    
    chart_type = "line"
    min_columns = 2
    
    def plot(self) -> None:
        """Generate the line chart."""
        if self.ax is None:
            self.create_figure()
        
        df = self._prepare_data()
        
        # Get X values
        if self.x_col and self.x_col in df.columns:
            x = df[self.x_col]
        else:
            x = np.arange(len(df))
        
        # Plot options
        alpha = self.options.get("alpha", 0.9)
        marker_size = self.options.get("marker_size", AcademicStyle.MARKER_SIZE)
        line_style = self.options.get("line_style", "solid")
        
        # Map line style string to matplotlib format
        line_style_map = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-."
        }
        base_ls = line_style_map.get(line_style, line_style)
        
        # Plot each Y column
        for i, y_col in enumerate(self.y_cols):
            if y_col not in df.columns:
                continue
            
            y = pd.to_numeric(df[y_col], errors="coerce")
            
            # Handle grouping
            if self.group_col and self.group_col in df.columns:
                groups = df[self.group_col].unique()
                for j, group in enumerate(groups):
                    mask = df[self.group_col] == group
                    idx = i * len(groups) + j
                    self.ax.plot(
                        x[mask], y[mask],
                        color=self._get_color(idx),
                        marker=self._get_marker(idx),
                        markersize=marker_size,
                        linestyle=base_ls,
                        linewidth=AcademicStyle.LINE_WIDTH,
                        alpha=alpha,
                        label=f"{y_col} ({group})"
                    )
            else:
                self.ax.plot(
                    x, y,
                    color=self._get_color(i),
                    marker=self._get_marker(i),
                    markersize=marker_size,
                    linestyle=base_ls if len(self.y_cols) == 1 else self._get_line_style(i),
                    linewidth=AcademicStyle.LINE_WIDTH,
                    alpha=alpha,
                    label=y_col
                )
        
        # Handle datetime x-axis
        if self.x_col and self.x_col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[self.x_col]):
                self.fig.autofmt_xdate()
