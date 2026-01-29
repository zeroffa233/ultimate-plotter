"""
UltimatePlotter (utp) - Academic-style CSV data visualization command-line tool.

A Python command-line tool that takes CSV files as input and outputs
high-quality charts with a unified academic style.
"""

__version__ = "1.0.0"
__author__ = "UltimatePlotter Team"

from .auto_scale import AutoScale
from .data_loader import DataLoader
from .style import AcademicStyle

__all__ = [
    "__version__",
    "AcademicStyle",
    "DataLoader",
    "AutoScale",
]
