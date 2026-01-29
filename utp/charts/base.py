"""
Base chart class providing common functionality for all chart types.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

from ..style import AcademicStyle
from ..auto_scale import AutoScale


class BaseChart(ABC):
    """
    Abstract base class for all chart types.
    
    Provides common functionality for figure creation, styling,
    axis configuration, and output generation.
    """
    
    # Chart type identifier (override in subclasses)
    chart_type: str = "base"
    
    # Minimum required columns
    min_columns: int = 1
    
    def __init__(
        self,
        data: pd.DataFrame,
        x_col: Optional[str] = None,
        y_cols: Optional[List[str]] = None,
        group_col: Optional[str] = None,
        size_col: Optional[str] = None,
        color_col: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the chart.
        
        Args:
            data: DataFrame containing the data to plot
            x_col: Column name for X axis
            y_cols: List of column names for Y axis
            group_col: Column name for grouping/coloring
            size_col: Column name for size encoding
            color_col: Column name for color encoding
            **kwargs: Additional chart-specific options
        """
        self.data = data
        self.x_col = x_col
        self.y_cols = y_cols or []
        self.group_col = group_col
        self.size_col = size_col
        self.color_col = color_col
        self.options = kwargs
        
        # Figure and axes (created in create_figure)
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        
        # Apply style
        AcademicStyle.apply()
    
    def create_figure(
        self,
        figsize: Optional[Tuple[float, float]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create the figure and axes with academic styling.
        
        Args:
            figsize: Optional custom figure size (width, height) in inches
            
        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = self._compute_figsize()
        
        self.fig, self.ax = AcademicStyle.create_figure(figsize=figsize)
        return self.fig, self.ax
    
    def _compute_figsize(self) -> Tuple[float, float]:
        """Compute appropriate figure size based on data."""
        n_categories = 1
        n_series = len(self.y_cols) if self.y_cols else 1
        
        if self.x_col and self.x_col in self.data.columns:
            n_categories = self.data[self.x_col].nunique()
        
        return AutoScale.compute_figure_size(
            n_categories=n_categories,
            n_series=n_series,
            chart_type=self.chart_type,
            base_width=self.options.get("figsize", (7, 5))[0],
            base_height=self.options.get("figsize", (7, 5))[1]
        )
    
    @abstractmethod
    def plot(self) -> None:
        """
        Generate the plot. Must be implemented by subclasses.
        """
        pass
    
    def configure_axes(
        self,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        log_x: bool = False,
        log_y: bool = False,
        grid: bool = True,
        legend: bool = True
    ) -> None:
        """
        Configure axis labels, limits, and other properties.
        
        Args:
            title: Chart title
            xlabel: X axis label
            ylabel: Y axis label
            xlim: X axis limits (min, max)
            ylim: Y axis limits (min, max)
            log_x: Use logarithmic X axis
            log_y: Use logarithmic Y axis
            grid: Show grid
            legend: Show legend
        """
        if self.ax is None:
            return
        
        # Set title
        if title:
            self.ax.set_title(title, fontsize=AcademicStyle.FONT_SIZE_TITLE, pad=10)
        
        # Set axis labels
        if xlabel:
            self.ax.set_xlabel(xlabel, fontsize=AcademicStyle.FONT_SIZE_LABEL)
        elif self.x_col:
            self.ax.set_xlabel(self.x_col, fontsize=AcademicStyle.FONT_SIZE_LABEL)
        
        if ylabel:
            self.ax.set_ylabel(ylabel, fontsize=AcademicStyle.FONT_SIZE_LABEL)
        elif len(self.y_cols) == 1:
            self.ax.set_ylabel(self.y_cols[0], fontsize=AcademicStyle.FONT_SIZE_LABEL)
        
        # Set axis limits
        if xlim:
            self.ax.set_xlim(xlim)
        
        if ylim:
            self.ax.set_ylim(ylim)
        
        # Set log scale
        if log_x:
            self.ax.set_xscale("log")
        
        if log_y:
            self.ax.set_yscale("log")
        
        # Add grid
        if grid:
            AcademicStyle.add_grid(self.ax)
        
        # Add legend
        if legend and self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc="best", framealpha=0.9)
    
    def auto_adjust_labels(self) -> None:
        """Automatically adjust labels to prevent overlap."""
        if self.ax is None or self.fig is None:
            return
        
        # Get x tick labels
        labels = [t.get_text() for t in self.ax.get_xticklabels()]
        if not labels or all(not l for l in labels):
            return
        
        fig_width = self.fig.get_figwidth()
        should_rotate, angle = AutoScale.should_rotate_labels(
            labels, fig_width, AcademicStyle.FONT_SIZE_TICK
        )
        
        if should_rotate:
            self.ax.tick_params(axis="x", rotation=angle)
            if angle == 90:
                ha = "center"
            else:
                ha = "right"
            for label in self.ax.get_xticklabels():
                label.set_horizontalalignment(ha)
    
    def save(
        self,
        output_path: Union[str, Path],
        format: str = "pdf",
        dpi: int = 300,
        transparent: bool = False
    ) -> Path:
        """
        Save the figure to a file.
        
        Args:
            output_path: Output file path or directory
            format: Output format (pdf, png, svg, eps)
            dpi: Resolution for raster formats
            transparent: Use transparent background
            
        Returns:
            Path to the saved file
        """
        if self.fig is None:
            raise ValueError("No figure to save. Call plot() first.")
        
        output_path = Path(output_path)
        
        # If directory, generate filename
        if output_path.is_dir():
            output_path = output_path / f"output_{self.chart_type}.{format}"
        
        # Ensure correct extension
        if output_path.suffix.lower() != f".{format}":
            output_path = output_path.with_suffix(f".{format}")
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Adjust layout and save
        self.fig.tight_layout()
        self.fig.savefig(
            output_path,
            format=format,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
            facecolor="white" if not transparent else "none",
            edgecolor="none"
        )
        
        return output_path
    
    def show(self) -> None:
        """Display the figure interactively."""
        if self.fig is None:
            raise ValueError("No figure to show. Call plot() first.")
        plt.show()
    
    def close(self) -> None:
        """Close the figure and free resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def _get_color(self, index: int) -> str:
        """Get color for series by index."""
        return AcademicStyle.get_color(index)
    
    def _get_marker(self, index: int) -> str:
        """Get marker for series by index."""
        return AcademicStyle.get_marker(index)
    
    def _get_line_style(self, index: int) -> str:
        """Get line style by index."""
        return AcademicStyle.get_line_style(index)
    
    def _prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for plotting (handle missing values, etc.).
        
        Returns:
            Cleaned DataFrame
        """
        df = self.data.copy()
        
        # Convert x column to numeric/datetime if needed
        if self.x_col and self.x_col in df.columns:
            col = df[self.x_col]
            if col.dtype == object:
                # Try datetime conversion
                try:
                    df[self.x_col] = pd.to_datetime(col, errors="coerce")
                except Exception:
                    pass
        
        return df
    
    def generate(
        self,
        output_path: Optional[Union[str, Path]] = None,
        format: str = "pdf",
        dpi: int = 300,
        show: bool = False,
        **kwargs
    ) -> Optional[Path]:
        """
        Complete workflow: create figure, plot, configure, and save/show.
        
        Args:
            output_path: Path to save figure (None to not save)
            format: Output format
            dpi: Resolution
            show: Whether to display interactively
            **kwargs: Additional options for configure_axes
            
        Returns:
            Path to saved file if output_path provided, else None
        """
        # Extract figsize if provided
        figsize = self.options.get("figsize")
        
        # Create figure
        self.create_figure(figsize=figsize)
        
        # Generate plot
        self.plot()
        
        # Configure axes
        self.configure_axes(
            title=self.options.get("title"),
            xlabel=self.options.get("xlabel"),
            ylabel=self.options.get("ylabel"),
            xlim=self.options.get("xlim"),
            ylim=self.options.get("ylim"),
            log_x=self.options.get("log_x", False),
            log_y=self.options.get("log_y", False),
            grid=not self.options.get("no_grid", False),
            legend=not self.options.get("no_legend", False),
            **kwargs
        )
        
        # Auto-adjust labels
        self.auto_adjust_labels()
        
        # Save if path provided
        saved_path = None
        if output_path:
            saved_path = self.save(output_path, format=format, dpi=dpi)
        
        # Show if requested
        if show:
            self.show()
        
        return saved_path
