"""
Auto-scaling module for intelligent axis range and tick calculation.

This module provides functions for automatically determining appropriate
axis ranges, tick intervals, and label formatting based on data characteristics.
"""

import numpy as np
from typing import Tuple, Optional, List, Union
import math


class AutoScale:
    """
    Automatic scaling utilities for axes and figure dimensions.
    
    Provides intelligent calculation of axis limits, tick intervals,
    and figure sizes based on data characteristics.
    """
    
    # Nice numbers for tick intervals
    NICE_NUMBERS = [1, 2, 2.5, 5, 10]
    
    @classmethod
    def compute_axis_limits(
        cls,
        data: np.ndarray,
        padding: float = 0.05,
        include_zero: bool = False,
        log_scale: bool = False
    ) -> Tuple[float, float]:
        """
        Compute appropriate axis limits for the given data.
        
        Args:
            data: Array of data values
            padding: Fraction of range to add as padding (default 5%)
            include_zero: Whether to ensure zero is included in the range
            log_scale: Whether the axis will use log scale
            
        Returns:
            Tuple of (min_limit, max_limit)
        """
        # Handle empty or all-nan data
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return (0, 1)
        
        data_min = float(np.min(valid_data))
        data_max = float(np.max(valid_data))
        
        # Handle single value case
        if data_min == data_max:
            if data_min == 0:
                return (-1, 1)
            magnitude = abs(data_min)
            return (data_min - 0.1 * magnitude, data_max + 0.1 * magnitude)
        
        if log_scale:
            # For log scale, work with positive values only
            positive_data = valid_data[valid_data > 0]
            if len(positive_data) == 0:
                return (0.1, 10)
            data_min = float(np.min(positive_data))
            data_max = float(np.max(positive_data))
            
            # Extend to nice log bounds
            log_min = math.floor(math.log10(data_min))
            log_max = math.ceil(math.log10(data_max))
            return (10 ** log_min, 10 ** log_max)
        
        # Calculate range and padding
        data_range = data_max - data_min
        pad = data_range * padding
        
        limit_min = data_min - pad
        limit_max = data_max + pad
        
        # Include zero if requested
        if include_zero:
            if limit_min > 0:
                limit_min = 0
            elif limit_max < 0:
                limit_max = 0
        
        # Round to nice numbers
        limit_min = cls._nice_floor(limit_min, data_range)
        limit_max = cls._nice_ceil(limit_max, data_range)
        
        return (limit_min, limit_max)
    
    @classmethod
    def compute_tick_interval(
        cls,
        data_min: float,
        data_max: float,
        target_ticks: int = 5
    ) -> float:
        """
        Compute a nice tick interval for the given range.
        
        Args:
            data_min: Minimum value of axis
            data_max: Maximum value of axis
            target_ticks: Desired approximate number of ticks
            
        Returns:
            Nice tick interval value
        """
        if data_min >= data_max:
            return 1.0
        
        data_range = data_max - data_min
        rough_interval = data_range / target_ticks
        
        # Find the magnitude
        magnitude = 10 ** math.floor(math.log10(rough_interval))
        
        # Normalize to range [1, 10]
        normalized = rough_interval / magnitude
        
        # Find the closest nice number
        nice_interval = None
        min_diff = float('inf')
        for nice in cls.NICE_NUMBERS:
            diff = abs(normalized - nice)
            if diff < min_diff:
                min_diff = diff
                nice_interval = nice
        
        return nice_interval * magnitude
    
    @classmethod
    def compute_ticks(
        cls,
        data_min: float,
        data_max: float,
        target_ticks: int = 5
    ) -> np.ndarray:
        """
        Compute nice tick positions for the given range.
        
        Args:
            data_min: Minimum value of axis
            data_max: Maximum value of axis
            target_ticks: Desired approximate number of ticks
            
        Returns:
            Array of tick positions
        """
        interval = cls.compute_tick_interval(data_min, data_max, target_ticks)
        
        # Find the first tick at or below data_min
        first_tick = math.floor(data_min / interval) * interval
        
        # Generate ticks
        ticks = []
        tick = first_tick
        while tick <= data_max + interval * 0.001:  # Small epsilon for floating point
            if tick >= data_min - interval * 0.001:
                ticks.append(tick)
            tick += interval
        
        return np.array(ticks)
    
    @classmethod
    def _nice_floor(cls, value: float, data_range: float) -> float:
        """Round value down to a nice number."""
        if data_range == 0:
            return value
        
        # Determine precision based on data range
        magnitude = 10 ** math.floor(math.log10(data_range))
        return math.floor(value / magnitude) * magnitude
    
    @classmethod
    def _nice_ceil(cls, value: float, data_range: float) -> float:
        """Round value up to a nice number."""
        if data_range == 0:
            return value
        
        magnitude = 10 ** math.floor(math.log10(data_range))
        return math.ceil(value / magnitude) * magnitude
    
    @classmethod
    def compute_figure_size(
        cls,
        n_categories: int = 1,
        n_series: int = 1,
        chart_type: str = "line",
        base_width: float = 7.0,
        base_height: float = 5.0,
        max_width: float = 14.0,
        max_height: float = 10.0
    ) -> Tuple[float, float]:
        """
        Compute adaptive figure size based on data characteristics.
        
        Args:
            n_categories: Number of categories (for bar charts, box plots, etc.)
            n_series: Number of data series
            chart_type: Type of chart
            base_width: Base figure width in inches
            base_height: Base figure height in inches
            max_width: Maximum allowed width
            max_height: Maximum allowed height
            
        Returns:
            Tuple of (width, height) in inches
        """
        width = base_width
        height = base_height
        
        if chart_type in ["bar", "box", "violin"]:
            # Scale width based on number of categories
            if n_categories > 6:
                width = min(base_width * (n_categories / 6), max_width)
            
            # For grouped bars, also consider series count
            if n_series > 3:
                width = min(width * 1.2, max_width)
        
        elif chart_type == "heatmap":
            # Scale both dimensions for heatmaps
            aspect = n_categories / max(n_series, 1)
            if aspect > 2:
                width = min(base_width * 1.5, max_width)
            elif aspect < 0.5:
                height = min(base_height * 1.5, max_height)
        
        elif chart_type == "hist":
            # Histograms for multiple series may need more height
            if n_series > 3:
                height = min(base_height * 1.3, max_height)
        
        return (width, height)
    
    @classmethod
    def compute_bins(
        cls,
        data: np.ndarray,
        method: str = "auto"
    ) -> Union[int, str]:
        """
        Compute appropriate number of bins for histogram.
        
        Args:
            data: Array of data values
            method: Binning method ('auto', 'sturges', 'fd', 'scott', or integer)
            
        Returns:
            Number of bins or binning method string
        """
        valid_data = data[np.isfinite(data)]
        n = len(valid_data)
        
        if n == 0:
            return 10
        
        if method == "auto":
            # Use Freedman-Diaconis rule with bounds
            if n < 20:
                return max(5, int(np.sqrt(n)))
            
            q75, q25 = np.percentile(valid_data, [75, 25])
            iqr = q75 - q25
            
            if iqr == 0:
                return int(np.sqrt(n))
            
            bin_width = 2 * iqr / (n ** (1/3))
            data_range = np.max(valid_data) - np.min(valid_data)
            n_bins = int(np.ceil(data_range / bin_width))
            
            # Bound the number of bins
            return max(5, min(n_bins, 100))
        
        elif method == "sturges":
            return int(np.ceil(np.log2(n) + 1))
        
        elif method in ["fd", "scott"]:
            return method
        
        else:
            try:
                return int(method)
            except (ValueError, TypeError):
                return "auto"
    
    @classmethod
    def should_rotate_labels(
        cls,
        labels: List[str],
        fig_width: float,
        font_size: int = 9
    ) -> Tuple[bool, int]:
        """
        Determine if x-axis labels should be rotated.
        
        Args:
            labels: List of label strings
            fig_width: Figure width in inches
            font_size: Font size in points
            
        Returns:
            Tuple of (should_rotate, rotation_angle)
        """
        if not labels:
            return (False, 0)
        
        n_labels = len(labels)
        max_label_len = max(len(str(label)) for label in labels)
        
        # Estimate character width (rough approximation)
        char_width_inches = font_size / 72 * 0.6  # Approximate
        total_label_width = n_labels * max_label_len * char_width_inches
        
        available_width = fig_width * 0.8  # Account for margins
        
        if total_label_width > available_width:
            if total_label_width > available_width * 2:
                return (True, 90)  # Vertical for very crowded
            else:
                return (True, 45)  # Diagonal for moderately crowded
        
        return (False, 0)
    
    @classmethod
    def compute_label_skip(
        cls,
        n_labels: int,
        fig_width: float,
        max_labels: int = 20
    ) -> int:
        """
        Compute label skip interval for crowded axes.
        
        Args:
            n_labels: Total number of labels
            fig_width: Figure width in inches
            max_labels: Maximum desired number of visible labels
            
        Returns:
            Skip interval (1 = show all, 2 = show every other, etc.)
        """
        # Adjust max_labels based on figure width
        adjusted_max = int(max_labels * (fig_width / 7.0))
        adjusted_max = max(5, adjusted_max)
        
        if n_labels <= adjusted_max:
            return 1
        
        return int(np.ceil(n_labels / adjusted_max))
    
    @classmethod
    def format_tick_label(
        cls,
        value: float,
        precision: Optional[int] = None
    ) -> str:
        """
        Format a tick value as a clean label string.
        
        Args:
            value: Numeric value to format
            precision: Optional decimal precision
            
        Returns:
            Formatted string
        """
        if precision is not None:
            return f"{value:.{precision}f}"
        
        # Auto-determine precision
        if value == 0:
            return "0"
        
        abs_val = abs(value)
        
        # Very large or small numbers: use scientific notation
        if abs_val >= 1e6 or (abs_val < 1e-3 and abs_val > 0):
            return f"{value:.2e}"
        
        # Check if it's effectively an integer
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        
        # Determine appropriate decimal places
        if abs_val >= 100:
            return f"{value:.0f}"
        elif abs_val >= 10:
            return f"{value:.1f}"
        elif abs_val >= 1:
            return f"{value:.2f}"
        else:
            return f"{value:.3f}"
