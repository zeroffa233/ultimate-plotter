"""
Command-line interface for UltimatePlotter.

Provides argument parsing and the main entry point for the utp command.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

from . import __version__
from .data_loader import DataLoader, interactive_column_selection
from .style import AcademicStyle
from .charts import (
    get_chart_class,
    list_chart_types,
    CHART_TYPES,
)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: Optional list of arguments (for testing)
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="utp",
        description="UltimatePlotter - Academic-style CSV data visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  utp data.csv -t line                           Basic line chart
  utp data.csv -t scatter -x time -y value       Scatter plot with specified columns
  utp data.csv -t bar -x category -y count       Bar chart
  utp *.csv -t line -x date -y temp -f png       Batch processing
  utp data.csv -t hist -y values --bins 20       Histogram with custom bins
  utp data.csv -t box -y score -g group          Grouped box plot
  utp data.csv -t heatmap -x col -y row -c val   Heatmap from three columns

For more information, visit: https://github.com/ultimate-plotter/ultimate-plotter
        """
    )
    
    # Version
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    # Auxiliary commands
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List all supported chart types"
    )
    
    parser.add_argument(
        "--show-colors",
        action="store_true",
        help="Preview the color palette"
    )
    
    # Required arguments
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="Input CSV file(s)"
    )
    
    parser.add_argument(
        "-t", "--type",
        dest="chart_type",
        choices=list(CHART_TYPES.keys()),
        help="Chart type"
    )
    
    # Column specification
    col_group = parser.add_argument_group("Column Selection")
    col_group.add_argument(
        "-x", "--x-col",
        help="X-axis column (name or index)"
    )
    col_group.add_argument(
        "-y", "--y-col",
        nargs="+",
        help="Y-axis column(s) (names or indices)"
    )
    col_group.add_argument(
        "-g", "--group-col",
        help="Grouping column for color/legend"
    )
    col_group.add_argument(
        "-s", "--size-col",
        help="Size encoding column (for scatter)"
    )
    col_group.add_argument(
        "-c", "--color-col",
        help="Color encoding column"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output",
        help="Output directory or file path (default: current directory)"
    )
    output_group.add_argument(
        "-f", "--format",
        choices=["pdf", "png", "svg", "eps"],
        default="pdf",
        help="Output format (default: pdf)"
    )
    output_group.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution (default: 300)"
    )
    output_group.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 7 5)"
    )
    
    # Chart customization
    custom_group = parser.add_argument_group("Chart Customization")
    custom_group.add_argument(
        "--title",
        help="Chart title"
    )
    custom_group.add_argument(
        "--xlabel",
        help="X-axis label"
    )
    custom_group.add_argument(
        "--ylabel",
        help="Y-axis label"
    )
    custom_group.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="X-axis limits"
    )
    custom_group.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Y-axis limits"
    )
    custom_group.add_argument(
        "--log-x",
        action="store_true",
        help="Use logarithmic X-axis"
    )
    custom_group.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic Y-axis"
    )
    custom_group.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable grid"
    )
    custom_group.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable legend"
    )
    
    # Chart-specific options
    specific_group = parser.add_argument_group("Chart-Specific Options")
    specific_group.add_argument(
        "--stacked",
        action="store_true",
        help="Stacked mode (for bar, area)"
    )
    specific_group.add_argument(
        "--horizontal",
        action="store_true",
        help="Horizontal orientation (for bar)"
    )
    specific_group.add_argument(
        "--bins",
        default="auto",
        help="Number of histogram bins (default: auto)"
    )
    specific_group.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Transparency (0-1, default: 0.8)"
    )
    specific_group.add_argument(
        "--marker-size",
        type=float,
        help="Marker size"
    )
    specific_group.add_argument(
        "--line-style",
        choices=["solid", "dashed", "dotted", "dashdot"],
        default="solid",
        help="Line style (default: solid)"
    )
    
    # Interactive mode
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Enable interactive column selection"
    )
    
    # Display mode
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (don't save)"
    )
    
    return parser.parse_args(args)


def resolve_column_arg(arg: Optional[str]) -> Optional[str]:
    """
    Resolve a column argument (could be name or index string).
    
    Args:
        arg: Column argument string
        
    Returns:
        Resolved column identifier
    """
    if arg is None:
        return None
    
    # Try to parse as integer index
    try:
        return int(arg)
    except ValueError:
        return arg


def generate_output_path(
    input_path: Path,
    chart_type: str,
    output_arg: Optional[str],
    format: str
) -> Path:
    """
    Generate output file path based on input file and chart type.
    
    Args:
        input_path: Input CSV file path
        chart_type: Type of chart
        output_arg: User-specified output path/directory
        format: Output format
        
    Returns:
        Output file path
    """
    filename = f"{input_path.stem}_{chart_type}.{format}"
    
    if output_arg:
        output_path = Path(output_arg)
        if output_path.is_dir():
            return output_path / filename
        elif output_path.suffix:
            return output_path
        else:
            # Treat as directory, create if needed
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path / filename
    else:
        return Path.cwd() / filename


def process_file(
    filepath: Path,
    args: argparse.Namespace
) -> Optional[Path]:
    """
    Process a single CSV file and generate the chart.
    
    Args:
        filepath: Path to CSV file
        args: Parsed command-line arguments
        
    Returns:
        Path to generated file, or None if failed
    """
    print(f"\nProcessing: {filepath}")
    
    # Load data
    try:
        loader = DataLoader(filepath)
        df = loader.load()
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"  Error loading file: {e}")
        return None
    
    # Resolve columns
    x_col = resolve_column_arg(args.x_col)
    y_cols = None
    if args.y_col:
        y_cols = [resolve_column_arg(y) for y in args.y_col]
    
    group_col = resolve_column_arg(args.group_col)
    size_col = resolve_column_arg(args.size_col)
    color_col = resolve_column_arg(args.color_col)
    
    # Interactive column selection
    if args.interactive or (x_col is None and y_cols is None):
        if sys.stdin.isatty():
            selection = interactive_column_selection(
                df, loader.column_types, args.chart_type
            )
            x_col = selection.get("x")
            y_cols = selection.get("y") or None
            group_col = group_col or selection.get("group")
        else:
            # Auto-select if not interactive
            selection = loader.auto_select_columns(
                args.chart_type, x_col, y_cols, group_col
            )
            x_col = selection["x"]
            y_cols = selection["y"] or None
            group_col = group_col or selection["group"]
    
    # Resolve column indices to names
    def resolve_to_name(col):
        if col is None:
            return None
        if isinstance(col, int):
            return loader._resolve_column(col)
        return col
    
    x_col = resolve_to_name(x_col)
    if y_cols:
        y_cols = [resolve_to_name(y) for y in y_cols]
    group_col = resolve_to_name(group_col)
    size_col = resolve_to_name(size_col)
    color_col = resolve_to_name(color_col)
    
    print(f"  X: {x_col}, Y: {y_cols}, Group: {group_col}")
    
    # Build chart options
    options = {
        "title": args.title,
        "xlabel": args.xlabel,
        "ylabel": args.ylabel,
        "xlim": tuple(args.xlim) if args.xlim else None,
        "ylim": tuple(args.ylim) if args.ylim else None,
        "log_x": args.log_x,
        "log_y": args.log_y,
        "no_grid": args.no_grid,
        "no_legend": args.no_legend,
        "stacked": args.stacked,
        "horizontal": args.horizontal,
        "bins": args.bins,
        "alpha": args.alpha,
        "line_style": args.line_style,
    }
    
    if args.figsize:
        options["figsize"] = tuple(args.figsize)
    
    if args.marker_size:
        options["marker_size"] = args.marker_size
    
    # Create chart
    try:
        ChartClass = get_chart_class(args.chart_type)
        chart = ChartClass(
            data=df,
            x_col=x_col,
            y_cols=y_cols,
            group_col=group_col,
            size_col=size_col,
            color_col=color_col,
            **options
        )
        
        # Generate
        if args.show:
            chart.generate(show=True)
            return None
        else:
            output_path = generate_output_path(
                filepath, args.chart_type, args.output, args.format
            )
            saved_path = chart.generate(
                output_path=output_path,
                format=args.format,
                dpi=args.dpi
            )
            print(f"  Saved: {saved_path}")
            chart.close()
            return saved_path
            
    except Exception as e:
        print(f"  Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the utp command.
    
    Args:
        args: Optional list of arguments (for testing)
        
    Returns:
        Exit code (0 for success)
    """
    parsed = parse_args(args)
    
    # Handle auxiliary commands
    if parsed.list_types:
        list_chart_types()
        return 0
    
    if parsed.show_colors:
        AcademicStyle.apply()
        fig = AcademicStyle.preview_colors()
        plt.show()
        return 0
    
    # Validate required arguments
    if not parsed.csv_files:
        print("Error: No input files specified. Use -h for help.")
        return 1
    
    if not parsed.chart_type:
        print("Error: Chart type (-t) is required. Use --list-types to see options.")
        return 1
    
    # Process each file
    results = []
    for filepath_str in parsed.csv_files:
        filepath = Path(filepath_str)
        
        # Handle glob patterns (shell should expand, but just in case)
        if "*" in str(filepath):
            import glob
            matched = glob.glob(str(filepath))
            for match in matched:
                result = process_file(Path(match), parsed)
                if result:
                    results.append(result)
        else:
            if not filepath.exists():
                print(f"Error: File not found: {filepath}")
                continue
            result = process_file(filepath, parsed)
            if result:
                results.append(result)
    
    # Summary
    if results:
        print(f"\nGenerated {len(results)} chart(s):")
        for path in results:
            print(f"  {path}")
    
    return 0 if results or parsed.show else 1


if __name__ == "__main__":
    sys.exit(main())
