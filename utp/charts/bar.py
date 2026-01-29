"""
Bar chart implementation.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List

from .base import BaseChart
from ..style import AcademicStyle
from ..auto_scale import AutoScale


class BarChart(BaseChart):
    """
    Bar chart for comparing categorical data.
    
    Supports grouped bars, stacked bars, and horizontal orientation.
    """
    
    chart_type = "bar"
    min_columns = 2
    
    def plot(self) -> None:
        """Generate the bar chart."""
        if self.ax is None:
            self.create_figure()
        
        df = self._prepare_data()
        
        # Get categories
        if not self.x_col or self.x_col not in df.columns:
            raise ValueError("X column (categories) is required for bar chart")
        
        categories = df[self.x_col].astype(str).values
        n_categories = len(categories)
        
        # Options
        stacked = self.options.get("stacked", False)
        horizontal = self.options.get("horizontal", False)
        alpha = self.options.get("alpha", 0.85)
        
        # Y columns to plot
        y_cols = [col for col in self.y_cols if col in df.columns]
        if not y_cols:
            raise ValueError("At least one valid Y column is required")
        
        n_series = len(y_cols)
        
        # Calculate bar positions
        x = np.arange(n_categories)
        
        if stacked:
            # Stacked bars
            bottom = np.zeros(n_categories)
            
            for i, col in enumerate(y_cols):
                values = pd.to_numeric(df[col], errors="coerce").fillna(0).values
                
                if horizontal:
                    self.ax.barh(
                        x, values,
                        left=bottom,
                        color=self._get_color(i),
                        alpha=alpha,
                        label=col,
                        edgecolor="white",
                        linewidth=0.5
                    )
                else:
                    self.ax.bar(
                        x, values,
                        bottom=bottom,
                        color=self._get_color(i),
                        alpha=alpha,
                        label=col,
                        edgecolor="white",
                        linewidth=0.5
                    )
                
                bottom += values
        else:
            # Grouped bars
            width = 0.8 / n_series
            
            for i, col in enumerate(y_cols):
                values = pd.to_numeric(df[col], errors="coerce").fillna(0).values
                offset = (i - n_series / 2 + 0.5) * width
                
                if horizontal:
                    self.ax.barh(
                        x + offset, values,
                        height=width,
                        color=self._get_color(i),
                        alpha=alpha,
                        label=col,
                        edgecolor="white",
                        linewidth=0.5
                    )
                else:
                    self.ax.bar(
                        x + offset, values,
                        width=width,
                        color=self._get_color(i),
                        alpha=alpha,
                        label=col,
                        edgecolor="white",
                        linewidth=0.5
                    )
        
        # Set tick labels
        if horizontal:
            self.ax.set_yticks(x)
            self.ax.set_yticklabels(categories)
            
            # Auto-adjust if too many categories
            if n_categories > 20:
                skip = AutoScale.compute_label_skip(n_categories, self.fig.get_figheight())
                visible_ticks = x[::skip]
                visible_labels = categories[::skip]
                self.ax.set_yticks(visible_ticks)
                self.ax.set_yticklabels(visible_labels)
        else:
            self.ax.set_xticks(x)
            self.ax.set_xticklabels(categories)
            
            # Check for rotation
            should_rotate, angle = AutoScale.should_rotate_labels(
                categories.tolist(),
                self.fig.get_figwidth(),
                AcademicStyle.FONT_SIZE_TICK
            )
            
            if should_rotate:
                self.ax.tick_params(axis="x", rotation=angle)
                ha = "right" if angle != 90 else "center"
                for label in self.ax.get_xticklabels():
                    label.set_horizontalalignment(ha)
        
        # Swap axis labels if horizontal
        if horizontal:
            self.options["xlabel"], self.options["ylabel"] = \
                self.options.get("ylabel"), self.options.get("xlabel")
    
    def auto_adjust_labels(self) -> None:
        """Override to handle rotation already done in plot()."""
        # Rotation is handled in plot() for bar charts
        pass
