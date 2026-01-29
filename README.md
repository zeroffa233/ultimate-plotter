# UltimatePlotter (utp)

A Python command-line tool for creating publication-ready, academic-style visualizations from CSV data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Unified Academic Style**: Consistent, publication-ready appearance across all chart types
- **Multiple Chart Types**: Line, scatter, bar, histogram, box, heatmap, area, and violin plots
- **Smart Defaults**: Automatic column type detection, axis scaling, and label formatting
- **Batch Processing**: Process multiple CSV files with the same settings
- **High-Quality Output**: PDF, PNG, SVG, and EPS formats at 300 DPI by default
- **Flexible Column Selection**: Specify columns by name or index, or use interactive mode

## Installation

```bash
# Clone the repository
git clone https://github.com/ultimate-plotter/ultimate-plotter.git
cd ultimate-plotter

# Install with pip
pip install -e .

# Or install dependencies only
pip install matplotlib pandas numpy seaborn
```

## Quick Start

```bash
# Basic line chart
utp data.csv -t line

# Scatter plot with specific columns
utp data.csv -t scatter -x time -y value -g category

# Bar chart with stacked bars
utp sales.csv -t bar -x quarter -y revenue expenses --stacked

# Histogram with custom bins
utp measurements.csv -t hist -y values --bins 30

# Save as PNG at 600 DPI
utp results.csv -t line -x epoch -y loss -f png --dpi 600
```

## Supported Chart Types

| Type | Description | Use Case |
|------|-------------|----------|
| `line` | Line chart | Trends over continuous variables |
| `scatter` | Scatter plot | Correlation between two variables |
| `bar` | Bar chart | Categorical comparisons |
| `hist` | Histogram | Distribution of values |
| `box` | Box plot | Distribution statistics and outliers |
| `heatmap` | Heatmap | Matrix data, correlations |
| `area` | Area chart | Stacked trends |
| `violin` | Violin plot | Distribution shape |

## Command-Line Reference

### Basic Usage

```bash
utp <csv_file> [csv_file2 ...] -t <chart_type> [options]
```

### Column Selection

| Option | Description |
|--------|-------------|
| `-x, --x-col` | X-axis column (name or 0-based index) |
| `-y, --y-col` | Y-axis column(s), can specify multiple |
| `-g, --group-col` | Grouping column for colors/legends |
| `-s, --size-col` | Size encoding (scatter plots) |
| `-c, --color-col` | Color encoding (continuous or categorical) |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | Current dir | Output directory or file path |
| `-f, --format` | `pdf` | Output format: pdf, png, svg, eps |
| `--dpi` | `300` | Resolution for raster formats |
| `--figsize W H` | `7 5` | Figure size in inches |

### Chart Customization

| Option | Description |
|--------|-------------|
| `--title` | Chart title |
| `--xlabel` | X-axis label |
| `--ylabel` | Y-axis label |
| `--xlim MIN MAX` | X-axis limits |
| `--ylim MIN MAX` | Y-axis limits |
| `--log-x` | Logarithmic X-axis |
| `--log-y` | Logarithmic Y-axis |
| `--no-grid` | Disable grid lines |
| `--no-legend` | Disable legend |

### Chart-Specific Options

| Option | Applies To | Description |
|--------|------------|-------------|
| `--stacked` | bar, area | Enable stacking |
| `--horizontal` | bar | Horizontal bars |
| `--bins` | hist | Number of bins (default: auto) |
| `--alpha` | all | Transparency (0-1) |
| `--marker-size` | line, scatter | Marker size |
| `--line-style` | line | solid, dashed, dotted, dashdot |

### Utility Commands

```bash
utp --help           # Show help message
utp --version        # Show version
utp --list-types     # List chart types with descriptions
utp --show-colors    # Preview the color palette
```

## Examples

### Time Series with Multiple Variables

```bash
utp weather.csv -t line -x date -y temperature humidity pressure \
    --title "Weather Data 2024" --ylabel "Value"
```

### Grouped Bar Chart

```bash
utp sales.csv -t bar -x product -y q1 q2 q3 q4 \
    --title "Quarterly Sales" -f png
```

### Correlation Heatmap

```bash
utp features.csv -t heatmap \
    --title "Feature Correlation Matrix"
```

### Distribution Comparison

```bash
utp experiment.csv -t violin -y score -g treatment \
    --title "Score Distribution by Treatment"
```

### Batch Processing

```bash
# Process all CSV files with the same settings
utp exp_*.csv -t line -x time -y measurement \
    -o results/ -f pdf --dpi 600
```

## Academic Style

UltimatePlotter applies a consistent academic style to all charts:

- **Color Palette**: 10-color palette optimized for both print and digital
- **Typography**: Sans-serif fonts (Arial/Helvetica) with appropriate sizes
- **Grid**: Subtle dotted grid lines for readability
- **Axes**: Clean spines with no top/right borders
- **Markers**: Consistent marker set that cycles for multiple series
- **Output**: 300 DPI default, suitable for journal submission

## Data Format

- CSV files with header row (column names in first row)
- UTF-8 encoding (with fallback to other encodings)
- Automatic type inference (numeric, categorical, datetime)
- Missing values handled automatically

## Python API

You can also use UltimatePlotter as a Python library:

```python
import pandas as pd
from utp.data_loader import DataLoader
from utp.charts import LineChart
from utp.style import AcademicStyle

# Load data
loader = DataLoader("data.csv")
df = loader.load()

# Create chart
chart = LineChart(
    data=df,
    x_col="time",
    y_cols=["value1", "value2"],
    title="My Chart"
)

# Generate and save
chart.generate(output_path="output.pdf", format="pdf", dpi=300)
```

## Requirements

- Python 3.8+
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- seaborn >= 0.11.0 (optional, for enhanced styling)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### v1.0.0

- Initial release
- Support for 8 chart types
- Unified academic styling
- Batch processing
- Interactive column selection
