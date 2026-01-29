"""
Chart type implementations.

This package contains individual chart type implementations,
each providing a consistent interface for data visualization.
"""

from .area import AreaChart
from .bar import BarChart
from .base import BaseChart
from .box import BoxChart
from .heatmap import HeatmapChart
from .hist import HistChart
from .line import LineChart
from .scatter import ScatterChart
from .violin import ViolinChart

# Registry of available chart types
CHART_TYPES = {
    "line": LineChart,
    "scatter": ScatterChart,
    "bar": BarChart,
    "hist": HistChart,
    "box": BoxChart,
    "heatmap": HeatmapChart,
    "area": AreaChart,
    "violin": ViolinChart,
}

CHART_DESCRIPTIONS = {
    "line": "折线图 - 用于展示连续变量的趋势变化",
    "scatter": "散点图 - 用于展示两个变量之间的相关性",
    "bar": "柱状图 - 用于分类数据的比较",
    "hist": "直方图 - 用于展示数值分布",
    "box": "箱线图 - 用于展示数据分布和异常值",
    "heatmap": "热力图 - 用于展示矩阵数据或相关性",
    "area": "面积图 - 用于展示堆叠趋势",
    "violin": "小提琴图 - 用于展示数据分布形态",
}


def get_chart_class(chart_type: str) -> type:
    """
    Get the chart class for a given chart type.

    Args:
        chart_type: Name of the chart type

    Returns:
        Chart class

    Raises:
        ValueError: If chart type is not supported
    """
    if chart_type not in CHART_TYPES:
        supported = ", ".join(CHART_TYPES.keys())
        raise ValueError(
            f"Unsupported chart type: {chart_type}. Supported types: {supported}"
        )
    return CHART_TYPES[chart_type]


def list_chart_types() -> None:
    """Print list of available chart types with descriptions."""
    print("\n支持的图表类型：")
    print("-" * 60)
    for name, desc in CHART_DESCRIPTIONS.items():
        print(f"  {name:12} : {desc}")
    print("-" * 60)


__all__ = [
    "BaseChart",
    "LineChart",
    "ScatterChart",
    "BarChart",
    "HistChart",
    "BoxChart",
    "HeatmapChart",
    "AreaChart",
    "ViolinChart",
    "CHART_TYPES",
    "CHART_DESCRIPTIONS",
    "get_chart_class",
    "list_chart_types",
]
