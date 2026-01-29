"""
Heatmap implementation.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from .base import BaseChart
from ..style import AcademicStyle


class HeatmapChart(BaseChart):
    """
    Heatmap for displaying matrix data or correlations.
    
    Can be created from:
    - Three columns (row, column, value)
    - A matrix-shaped DataFrame
    """
    
    chart_type = "heatmap"
    min_columns = 3
    
    def plot(self) -> None:
        """Generate the heatmap."""
        if self.ax is None:
            self.create_figure()
        
        df = self._prepare_data()
        
        # Determine data format and create matrix
        matrix, row_labels, col_labels = self._prepare_matrix(df)
        
        # Options
        alpha = self.options.get("alpha", 1.0)
        cmap = self.options.get("cmap", "RdYlBu_r")  # Diverging colormap
        
        # Create heatmap
        im = self.ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            alpha=alpha
        )
        
        # Add colorbar
        cbar = self.fig.colorbar(im, ax=self.ax, pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=AcademicStyle.FONT_SIZE_TICK)
        
        # Set ticks and labels
        self.ax.set_xticks(np.arange(len(col_labels)))
        self.ax.set_yticks(np.arange(len(row_labels)))
        self.ax.set_xticklabels(col_labels)
        self.ax.set_yticklabels(row_labels)
        
        # Rotate x labels if needed
        if len(col_labels) > 8 or max(len(str(l)) for l in col_labels) > 8:
            self.ax.tick_params(axis="x", rotation=45)
            for label in self.ax.get_xticklabels():
                label.set_horizontalalignment("right")
        
        # Add cell annotations if matrix is small enough
        if matrix.shape[0] * matrix.shape[1] <= 100:
            self._add_annotations(matrix)
        
        # Set spines visible for heatmap
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    def _prepare_matrix(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, list, list]:
        """
        Prepare matrix data from DataFrame.
        
        Returns:
            Tuple of (matrix, row_labels, column_labels)
        """
        # Check if data is already matrix-like (no explicit row/col columns)
        if self.x_col is None and self.y_cols is None:
            # Treat DataFrame as matrix
            matrix = df.select_dtypes(include=[np.number]).values
            row_labels = df.index.tolist()
            col_labels = df.select_dtypes(include=[np.number]).columns.tolist()
            return matrix, row_labels, col_labels
        
        # Three-column format: row, col, value
        if self.x_col and self.y_cols and len(self.y_cols) >= 1:
            row_col = self.y_cols[0]  # Y column is row index
            col_col = self.x_col  # X column is column index
            
            # Value column: use group_col or the third column
            if self.group_col and self.group_col in df.columns:
                val_col = self.group_col
            elif len(self.y_cols) >= 2 and self.y_cols[1] in df.columns:
                val_col = self.y_cols[1]
            else:
                # Find first numeric column not used
                used = {row_col, col_col}
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                val_col = None
                for c in numeric_cols:
                    if c not in used:
                        val_col = c
                        break
                
                if val_col is None:
                    raise ValueError("Cannot determine value column for heatmap")
            
            # Pivot the data
            pivot_df = df.pivot_table(
                index=row_col,
                columns=col_col,
                values=val_col,
                aggfunc="mean"
            )
            
            matrix = pivot_df.values
            row_labels = pivot_df.index.tolist()
            col_labels = pivot_df.columns.tolist()
            
            return matrix, row_labels, col_labels
        
        # Fallback: correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr = numeric_df.corr()
            return corr.values, corr.index.tolist(), corr.columns.tolist()
        
        raise ValueError("Cannot create heatmap from provided data")
    
    def _add_annotations(self, matrix: np.ndarray) -> None:
        """Add value annotations to cells."""
        # Determine text color based on cell value
        norm = plt.Normalize(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                
                # Choose text color for contrast
                normalized_val = norm(value)
                text_color = "white" if 0.3 < normalized_val < 0.7 else "black"
                
                # Format value
                if abs(value) >= 100:
                    text = f"{value:.0f}"
                elif abs(value) >= 10:
                    text = f"{value:.1f}"
                else:
                    text = f"{value:.2f}"
                
                self.ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color=text_color,
                    fontsize=AcademicStyle.FONT_SIZE_TICK - 1,
                    fontweight="normal"
                )
    
    def configure_axes(self, **kwargs) -> None:
        """Override to handle heatmap-specific axis configuration."""
        # Don't show grid for heatmap
        kwargs["grid"] = False
        
        # Call parent configuration
        super().configure_axes(**kwargs)
        
        # Keep all spines for heatmap
        if self.ax:
            for spine in self.ax.spines.values():
                spine.set_visible(True)
