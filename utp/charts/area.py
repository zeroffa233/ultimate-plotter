"""
Area chart implementation.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List

from .base import BaseChart
from ..style import AcademicStyle


class AreaChart(BaseChart):
    """
    Area chart for displaying stacked trends over time.
    
    Similar to line charts but with filled areas between
    the line and the x-axis or between series.
    """
    
    chart_type = "area"
    min_columns = 2
    
    def plot(self) -> None:
        """Generate the area chart."""
        if self.ax is None:
            self.create_figure()
        
        df = self._prepare_data()
        
        # Get X values
        if self.x_col and self.x_col in df.columns:
            x = df[self.x_col]
        else:
            x = np.arange(len(df))
        
        # Get Y columns
        y_cols = [col for col in self.y_cols if col in df.columns]
        if not y_cols:
            raise ValueError("At least one valid Y column is required")
        
        # Options
        stacked = self.options.get("stacked", True)  # Default stacked for area
        alpha = self.options.get("alpha", 0.7)
        
        # Prepare Y data
        y_data = []
        for col in y_cols:
            y_data.append(pd.to_numeric(df[col], errors="coerce").fillna(0).values)
        
        y_data = np.array(y_data)
        
        colors = AcademicStyle.get_colors(len(y_cols))
        
        if stacked:
            # Stacked area chart
            self.ax.stackplot(
                x, y_data,
                labels=y_cols,
                colors=colors,
                alpha=alpha,
                edgecolor="white",
                linewidth=0.5
            )
        else:
            # Overlapping areas
            for i, (col, y) in enumerate(zip(y_cols, y_data)):
                self.ax.fill_between(
                    x, y,
                    color=colors[i],
                    alpha=alpha * 0.5,  # Reduce alpha for overlap
                    label=col,
                    edgecolor=colors[i],
                    linewidth=1
                )
                # Add line on top
                self.ax.plot(
                    x, y,
                    color=colors[i],
                    linewidth=AcademicStyle.LINE_WIDTH,
                    alpha=0.9
                )
        
        # Handle datetime x-axis
        if self.x_col and self.x_col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[self.x_col]):
                self.fig.autofmt_xdate()
        
        # Set Y limit to start at 0 for stacked
        if stacked:
            self.ax.set_ylim(bottom=0)
