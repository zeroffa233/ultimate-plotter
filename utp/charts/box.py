"""
Box plot implementation.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from .base import BaseChart
from ..style import AcademicStyle


class BoxChart(BaseChart):
    """
    Box plot for displaying distribution statistics.
    
    Shows median, quartiles, and outliers for one or more numeric variables,
    optionally grouped by a categorical variable.
    """
    
    chart_type = "box"
    min_columns = 1
    
    def plot(self) -> None:
        """Generate the box plot."""
        if self.ax is None:
            self.create_figure()
        
        df = self._prepare_data()
        
        # Get columns to plot
        if self.y_cols:
            columns = [col for col in self.y_cols if col in df.columns]
        else:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            raise ValueError("No numeric columns found for box plot")
        
        # Options
        alpha = self.options.get("alpha", 0.8)
        
        # Check for grouping
        if self.group_col and self.group_col in df.columns:
            self._plot_grouped(df, columns, alpha)
        else:
            self._plot_simple(df, columns, alpha)
    
    def _plot_simple(
        self,
        df: pd.DataFrame,
        columns: List[str],
        alpha: float
    ) -> None:
        """Plot box plots for multiple columns without grouping."""
        # Prepare data
        data = []
        labels = []
        
        for col in columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(values) > 0:
                data.append(values)
                labels.append(col)
        
        if not data:
            raise ValueError("No valid numeric data for box plot")
        
        # Create box plot
        bp = self.ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=False,
            widths=0.6
        )
        
        # Style the boxes
        self._style_boxplot(bp, len(data), alpha)
        
        # Set ylabel
        if not self.options.get("ylabel"):
            self.options["ylabel"] = "Value"
    
    def _plot_grouped(
        self,
        df: pd.DataFrame,
        columns: List[str],
        alpha: float
    ) -> None:
        """Plot grouped box plots."""
        groups = df[self.group_col].unique()
        n_groups = len(groups)
        n_cols = len(columns)
        
        # Prepare data
        data = []
        positions = []
        labels = []
        colors = []
        
        width = 0.8 / n_cols
        
        for i, col in enumerate(columns):
            for j, group in enumerate(groups):
                mask = df[self.group_col] == group
                values = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna().values
                
                if len(values) > 0:
                    data.append(values)
                    pos = j + (i - n_cols / 2 + 0.5) * width
                    positions.append(pos)
                    colors.append(self._get_color(i))
        
        if not data:
            raise ValueError("No valid numeric data for box plot")
        
        # Create box plot
        bp = self.ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            notch=False,
            widths=width * 0.8
        )
        
        # Style with colors
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)
        
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(1)
        
        for line in bp["medians"]:
            line.set_color("black")
            line.set_linewidth(1.5)
        
        for flier in bp["fliers"]:
            flier.set_marker("o")
            flier.set_markersize(4)
            flier.set_markerfacecolor("none")
            flier.set_markeredgecolor("gray")
        
        # Set x-axis labels
        self.ax.set_xticks(range(n_groups))
        self.ax.set_xticklabels([str(g) for g in groups])
        
        # Create legend
        if n_cols > 1:
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=self._get_color(i), alpha=alpha)
                for i in range(n_cols)
            ]
            self.ax.legend(legend_handles, columns, loc="best")
        
        # Set labels
        if not self.options.get("xlabel"):
            self.options["xlabel"] = self.group_col
        if not self.options.get("ylabel"):
            self.options["ylabel"] = "Value" if n_cols > 1 else columns[0]
    
    def _style_boxplot(
        self,
        bp: Dict[str, Any],
        n_boxes: int,
        alpha: float
    ) -> None:
        """Apply academic styling to box plot elements."""
        colors = AcademicStyle.get_colors(n_boxes)
        
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)
        
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(1)
        
        for line in bp["medians"]:
            line.set_color("black")
            line.set_linewidth(1.5)
        
        for flier in bp["fliers"]:
            flier.set_marker("o")
            flier.set_markersize(4)
            flier.set_markerfacecolor("none")
            flier.set_markeredgecolor("gray")
