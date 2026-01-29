"""
Violin plot implementation.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List

from .base import BaseChart
from ..style import AcademicStyle


class ViolinChart(BaseChart):
    """
    Violin plot for displaying distribution shapes.
    
    Combines box plot statistics with kernel density estimation
    to show the full distribution shape.
    """
    
    chart_type = "violin"
    min_columns = 1
    
    def plot(self) -> None:
        """Generate the violin plot."""
        if self.ax is None:
            self.create_figure()
        
        df = self._prepare_data()
        
        # Get columns to plot
        if self.y_cols:
            columns = [col for col in self.y_cols if col in df.columns]
        else:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            raise ValueError("No numeric columns found for violin plot")
        
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
        """Plot violin plots for multiple columns without grouping."""
        # Prepare data
        data = []
        labels = []
        
        for col in columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(values) > 1:  # Need at least 2 points for KDE
                data.append(values)
                labels.append(col)
        
        if not data:
            raise ValueError("No valid numeric data for violin plot")
        
        # Create violin plot
        positions = range(1, len(data) + 1)
        vp = self.ax.violinplot(
            data,
            positions=positions,
            showmeans=False,
            showmedians=True,
            showextrema=True
        )
        
        # Style the violins
        colors = AcademicStyle.get_colors(len(data))
        
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i])
            body.set_edgecolor("black")
            body.set_alpha(alpha)
            body.set_linewidth(0.5)
        
        # Style lines
        for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if partname in vp:
                vp[partname].set_edgecolor("black")
                vp[partname].set_linewidth(1)
        
        # Add quartile lines (like box plot)
        self._add_quartile_indicators(data, positions, colors, alpha)
        
        # Set labels
        self.ax.set_xticks(positions)
        self.ax.set_xticklabels(labels)
        
        if not self.options.get("ylabel"):
            self.options["ylabel"] = "Value"
    
    def _plot_grouped(
        self,
        df: pd.DataFrame,
        columns: List[str],
        alpha: float
    ) -> None:
        """Plot grouped violin plots."""
        groups = df[self.group_col].unique()
        n_groups = len(groups)
        n_cols = len(columns)
        
        # Calculate positions
        width = 0.8 / n_cols
        
        all_data = []
        all_positions = []
        all_colors = []
        
        for i, col in enumerate(columns):
            for j, group in enumerate(groups):
                mask = df[self.group_col] == group
                values = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna().values
                
                if len(values) > 1:
                    pos = j + (i - n_cols / 2 + 0.5) * width
                    all_data.append(values)
                    all_positions.append(pos)
                    all_colors.append(self._get_color(i))
        
        if not all_data:
            raise ValueError("No valid numeric data for violin plot")
        
        # Create violin plot
        vp = self.ax.violinplot(
            all_data,
            positions=all_positions,
            widths=width * 0.9,
            showmeans=False,
            showmedians=True,
            showextrema=True
        )
        
        # Style the violins with colors
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(all_colors[i])
            body.set_edgecolor("black")
            body.set_alpha(alpha)
            body.set_linewidth(0.5)
        
        # Style lines
        for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if partname in vp:
                vp[partname].set_edgecolor("black")
                vp[partname].set_linewidth(1)
        
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
    
    def _add_quartile_indicators(
        self,
        data: List[np.ndarray],
        positions: range,
        colors: List[str],
        alpha: float
    ) -> None:
        """Add quartile indicator boxes inside violins."""
        for i, (d, pos) in enumerate(zip(data, positions)):
            q1, median, q3 = np.percentile(d, [25, 50, 75])
            
            # Draw small box for IQR
            box_width = 0.1
            self.ax.fill_betweenx(
                [q1, q3],
                pos - box_width,
                pos + box_width,
                color="white",
                alpha=0.9,
                zorder=2
            )
            self.ax.hlines(median, pos - box_width, pos + box_width,
                          color="black", linewidth=2, zorder=3)
