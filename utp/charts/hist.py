"""
Histogram implementation.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Union

from .base import BaseChart
from ..style import AcademicStyle
from ..auto_scale import AutoScale


class HistChart(BaseChart):
    """
    Histogram for displaying the distribution of numeric values.
    
    Supports multiple series, automatic bin calculation,
    and various display options.
    """
    
    chart_type = "hist"
    min_columns = 1
    
    def plot(self) -> None:
        """Generate the histogram."""
        if self.ax is None:
            self.create_figure()
        
        df = self._prepare_data()
        
        # Get columns to plot
        if self.y_cols:
            columns = [col for col in self.y_cols if col in df.columns]
        else:
            # Use all numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            raise ValueError("No numeric columns found for histogram")
        
        # Options
        alpha = self.options.get("alpha", 0.7)
        bins = self.options.get("bins", "auto")
        stacked = self.options.get("stacked", False)
        
        # Collect data for all columns
        data_arrays = []
        labels = []
        colors = []
        
        for i, col in enumerate(columns):
            values = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(values) > 0:
                data_arrays.append(values)
                labels.append(col)
                colors.append(self._get_color(i))
        
        if not data_arrays:
            raise ValueError("No valid numeric data for histogram")
        
        # Calculate bins
        if bins == "auto":
            # Use combined data range for consistent bins
            all_data = np.concatenate(data_arrays)
            n_bins = AutoScale.compute_bins(all_data, method="auto")
        else:
            n_bins = int(bins) if str(bins).isdigit() else bins
        
        # Plot histogram(s)
        if len(data_arrays) == 1:
            # Single histogram
            self.ax.hist(
                data_arrays[0],
                bins=n_bins,
                color=colors[0],
                alpha=alpha,
                edgecolor="white",
                linewidth=0.5,
                label=labels[0]
            )
        else:
            # Multiple histograms
            if stacked:
                self.ax.hist(
                    data_arrays,
                    bins=n_bins,
                    color=colors,
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.5,
                    label=labels,
                    stacked=True
                )
            else:
                # Overlapping with reduced alpha
                adjusted_alpha = min(alpha, 0.5)
                for data, label, color in zip(data_arrays, labels, colors):
                    self.ax.hist(
                        data,
                        bins=n_bins,
                        color=color,
                        alpha=adjusted_alpha,
                        edgecolor="white",
                        linewidth=0.5,
                        label=label
                    )
        
        # Set labels
        if not self.options.get("xlabel"):
            if len(columns) == 1:
                self.options["xlabel"] = columns[0]
            else:
                self.options["xlabel"] = "Value"
        
        if not self.options.get("ylabel"):
            self.options["ylabel"] = "Frequency"
