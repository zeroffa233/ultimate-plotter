"""
Unified academic style configuration for all charts.

This module defines the visual style parameters that ensure all charts
have a consistent, publication-ready appearance.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple, Optional
import numpy as np


class AcademicStyle:
    """
    Unified academic style configuration.
    
    Provides consistent styling for all chart types including colors,
    markers, fonts, line widths, and other visual parameters.
    """
    
    # Academic-friendly color palette (based on Tableau10 with modifications)
    COLORS: List[str] = [
        "#4E79A7",  # Steel Blue
        "#F28E2B",  # Orange
        "#E15759",  # Red
        "#76B7B2",  # Teal
        "#59A14F",  # Green
        "#EDC948",  # Yellow
        "#B07AA1",  # Purple
        "#FF9DA7",  # Pink
        "#9C755F",  # Brown
        "#BAB0AC",  # Gray
    ]
    
    # Marker styles for line and scatter plots
    MARKERS: List[str] = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]
    
    # Line styles
    LINE_STYLES: List[str] = ["-", "--", "-.", ":"]
    
    # Default figure sizes (width, height in inches)
    FIGSIZE_SINGLE: Tuple[float, float] = (3.5, 2.5)  # Single column
    FIGSIZE_DOUBLE: Tuple[float, float] = (7.0, 5.0)  # Double column
    
    # Font settings
    FONT_FAMILY: str = "sans-serif"
    FONT_SANS_SERIF: List[str] = ["Arial", "Helvetica", "DejaVu Sans"]
    FONT_SIZE_TITLE: int = 12
    FONT_SIZE_LABEL: int = 10
    FONT_SIZE_TICK: int = 9
    FONT_SIZE_LEGEND: int = 9
    
    # Line and marker settings
    LINE_WIDTH: float = 1.5
    MARKER_SIZE: float = 6.0
    MARKER_EDGE_WIDTH: float = 0.5
    
    # Grid settings
    GRID_ALPHA: float = 0.3
    GRID_LINE_STYLE: str = "--"
    GRID_LINE_WIDTH: float = 0.5
    
    # Axes settings
    AXES_LINE_WIDTH: float = 1.0
    SPINE_COLOR: str = "#333333"
    
    # Default alpha for filled areas
    FILL_ALPHA: float = 0.8
    
    # Default DPI for output
    DEFAULT_DPI: int = 300
    
    @classmethod
    def apply(cls, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None) -> None:
        """
        Apply the academic style to the current matplotlib configuration.
        
        Args:
            fig: Optional figure to style
            ax: Optional axes to style
        """
        # Set global rcParams
        plt.rcParams.update({
            # Font settings
            "font.family": cls.FONT_FAMILY,
            "font.sans-serif": cls.FONT_SANS_SERIF,
            "font.size": cls.FONT_SIZE_TICK,
            
            # Axes settings
            "axes.titlesize": cls.FONT_SIZE_TITLE,
            "axes.labelsize": cls.FONT_SIZE_LABEL,
            "axes.linewidth": cls.AXES_LINE_WIDTH,
            "axes.edgecolor": cls.SPINE_COLOR,
            "axes.labelcolor": cls.SPINE_COLOR,
            "axes.prop_cycle": mpl.cycler(color=cls.COLORS),
            "axes.spines.top": False,
            "axes.spines.right": False,
            
            # Tick settings
            "xtick.labelsize": cls.FONT_SIZE_TICK,
            "ytick.labelsize": cls.FONT_SIZE_TICK,
            "xtick.color": cls.SPINE_COLOR,
            "ytick.color": cls.SPINE_COLOR,
            "xtick.direction": "out",
            "ytick.direction": "out",
            
            # Grid settings
            "grid.alpha": cls.GRID_ALPHA,
            "grid.linestyle": cls.GRID_LINE_STYLE,
            "grid.linewidth": cls.GRID_LINE_WIDTH,
            
            # Legend settings
            "legend.fontsize": cls.FONT_SIZE_LEGEND,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "legend.fancybox": False,
            
            # Figure settings
            "figure.dpi": 100,
            "savefig.dpi": cls.DEFAULT_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            
            # Line settings
            "lines.linewidth": cls.LINE_WIDTH,
            "lines.markersize": cls.MARKER_SIZE,
            "lines.markeredgewidth": cls.MARKER_EDGE_WIDTH,
            
            # Patch settings (for bars, etc.)
            "patch.linewidth": 0.5,
            "patch.edgecolor": "#333333",
            
            # Text settings
            "text.color": cls.SPINE_COLOR,
            
            # Math text
            "mathtext.fontset": "dejavusans",
        })
        
        # Apply to specific axes if provided
        if ax is not None:
            cls._style_axes(ax)
    
    @classmethod
    def _style_axes(cls, ax: plt.Axes) -> None:
        """Apply styling to specific axes."""
        # Set spine visibility and color
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(cls.AXES_LINE_WIDTH)
            ax.spines[spine].set_color(cls.SPINE_COLOR)
    
    @classmethod
    def get_color(cls, index: int) -> str:
        """Get color by index, cycling through the palette."""
        return cls.COLORS[index % len(cls.COLORS)]
    
    @classmethod
    def get_marker(cls, index: int) -> str:
        """Get marker by index, cycling through available markers."""
        return cls.MARKERS[index % len(cls.MARKERS)]
    
    @classmethod
    def get_line_style(cls, index: int) -> str:
        """Get line style by index, cycling through available styles."""
        return cls.LINE_STYLES[index % len(cls.LINE_STYLES)]
    
    @classmethod
    def get_colors(cls, n: int) -> List[str]:
        """Get n colors from the palette, cycling if necessary."""
        return [cls.get_color(i) for i in range(n)]
    
    @classmethod
    def get_markers(cls, n: int) -> List[str]:
        """Get n markers, cycling if necessary."""
        return [cls.get_marker(i) for i in range(n)]
    
    @classmethod
    def create_figure(
        cls,
        figsize: Optional[Tuple[float, float]] = None,
        single_column: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a new figure with academic styling applied.
        
        Args:
            figsize: Optional custom figure size (width, height) in inches
            single_column: If True, use single column size; otherwise double column
            
        Returns:
            Tuple of (figure, axes)
        """
        cls.apply()
        
        if figsize is None:
            figsize = cls.FIGSIZE_SINGLE if single_column else cls.FIGSIZE_DOUBLE
        
        fig, ax = plt.subplots(figsize=figsize)
        cls._style_axes(ax)
        
        return fig, ax
    
    @classmethod
    def add_grid(cls, ax: plt.Axes, which: str = "major", axis: str = "both") -> None:
        """
        Add styled grid to axes.
        
        Args:
            ax: Axes to add grid to
            which: Which grid lines ('major', 'minor', 'both')
            axis: Which axis ('x', 'y', 'both')
        """
        ax.grid(
            True,
            which=which,
            axis=axis,
            alpha=cls.GRID_ALPHA,
            linestyle=cls.GRID_LINE_STYLE,
            linewidth=cls.GRID_LINE_WIDTH
        )
        ax.set_axisbelow(True)
    
    @classmethod
    def preview_colors(cls) -> plt.Figure:
        """
        Generate a preview figure showing the color palette.
        
        Returns:
            Figure showing color swatches with hex codes
        """
        cls.apply()
        
        n_colors = len(cls.COLORS)
        fig, ax = plt.subplots(figsize=(8, 2))
        
        for i, color in enumerate(cls.COLORS):
            ax.barh(0, 1, left=i, color=color, edgecolor="white", linewidth=1)
            ax.text(i + 0.5, 0, color, ha="center", va="center", 
                   fontsize=8, color="white" if i < 5 else "black",
                   fontweight="bold")
        
        ax.set_xlim(0, n_colors)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        ax.set_title("UltimatePlotter Academic Color Palette", fontsize=12, pad=10)
        
        plt.tight_layout()
        return fig
    
    @classmethod
    def preview_markers(cls) -> plt.Figure:
        """
        Generate a preview figure showing available markers.
        
        Returns:
            Figure showing marker styles
        """
        cls.apply()
        
        n_markers = len(cls.MARKERS)
        fig, ax = plt.subplots(figsize=(8, 2))
        
        x = np.arange(n_markers)
        for i, marker in enumerate(cls.MARKERS):
            ax.scatter(i, 0, marker=marker, s=150, c=cls.COLORS[i],
                      edgecolors="black", linewidths=0.5)
            ax.text(i, -0.3, f"'{marker}'", ha="center", va="top", fontsize=9)
        
        ax.set_xlim(-0.5, n_markers - 0.5)
        ax.set_ylim(-0.6, 0.4)
        ax.axis("off")
        ax.set_title("UltimatePlotter Marker Styles", fontsize=12, pad=10)
        
        plt.tight_layout()
        return fig
