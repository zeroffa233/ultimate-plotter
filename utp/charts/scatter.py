"""
Scatter chart implementation.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

from .base import BaseChart
from ..style import AcademicStyle


class ScatterChart(BaseChart):
    """
    Scatter chart for displaying relationships between two variables.
    
    Supports optional color encoding, size encoding, and grouping
    for multi-dimensional data visualization.
    """
    
    chart_type = "scatter"
    min_columns = 2
    
    def plot(self) -> None:
        """Generate the scatter chart."""
        if self.ax is None:
            self.create_figure()
        
        df = self._prepare_data()
        
        # Get X and Y values
        if not self.x_col or self.x_col not in df.columns:
            raise ValueError("X column is required for scatter plot")
        
        if not self.y_cols or self.y_cols[0] not in df.columns:
            raise ValueError("Y column is required for scatter plot")
        
        x = pd.to_numeric(df[self.x_col], errors="coerce")
        y = pd.to_numeric(df[self.y_cols[0]], errors="coerce")
        
        # Options
        alpha = self.options.get("alpha", 0.7)
        base_size = self.options.get("marker_size", AcademicStyle.MARKER_SIZE * 10)
        
        # Size encoding
        sizes = None
        if self.size_col and self.size_col in df.columns:
            size_data = pd.to_numeric(df[self.size_col], errors="coerce")
            # Normalize to reasonable marker sizes
            size_min, size_max = size_data.min(), size_data.max()
            if size_max > size_min:
                sizes = 20 + (size_data - size_min) / (size_max - size_min) * 200
            else:
                sizes = base_size
        else:
            sizes = base_size
        
        # Color encoding or grouping
        if self.group_col and self.group_col in df.columns:
            groups = df[self.group_col].unique()
            
            for i, group in enumerate(groups):
                mask = df[self.group_col] == group
                
                scatter = self.ax.scatter(
                    x[mask], y[mask],
                    c=self._get_color(i),
                    s=sizes[mask] if hasattr(sizes, '__iter__') else sizes,
                    marker=self._get_marker(i),
                    alpha=alpha,
                    edgecolors="white",
                    linewidths=0.5,
                    label=str(group)
                )
        
        elif self.color_col and self.color_col in df.columns:
            # Continuous color mapping
            color_data = df[self.color_col]
            
            if pd.api.types.is_numeric_dtype(color_data):
                # Continuous colormap
                scatter = self.ax.scatter(
                    x, y,
                    c=color_data,
                    s=sizes,
                    cmap="viridis",
                    alpha=alpha,
                    edgecolors="white",
                    linewidths=0.5
                )
                cbar = self.fig.colorbar(scatter, ax=self.ax, pad=0.02)
                cbar.set_label(self.color_col, fontsize=AcademicStyle.FONT_SIZE_LABEL)
            else:
                # Categorical color
                categories = color_data.unique()
                for i, cat in enumerate(categories):
                    mask = color_data == cat
                    self.ax.scatter(
                        x[mask], y[mask],
                        c=self._get_color(i),
                        s=sizes[mask] if hasattr(sizes, '__iter__') else sizes,
                        marker=self._get_marker(i),
                        alpha=alpha,
                        edgecolors="white",
                        linewidths=0.5,
                        label=str(cat)
                    )
        else:
            # Simple scatter
            self.ax.scatter(
                x, y,
                c=self._get_color(0),
                s=sizes,
                marker=self._get_marker(0),
                alpha=alpha,
                edgecolors="white",
                linewidths=0.5,
                label=self.y_cols[0]
            )
        
        # Set labels
        if not self.options.get("xlabel"):
            self.options["xlabel"] = self.x_col
        if not self.options.get("ylabel"):
            self.options["ylabel"] = self.y_cols[0]
